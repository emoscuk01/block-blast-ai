"""
Block Blast AI — MaskablePPO Eğitim Scripti

Not (5090 / Vast): Ortam simülasyonu CPU Python'da; GPU yalnızca PPO gradient adımlarında çalışır.
CUDA algilandiginda otomatik olarak daha genis MLP + buyuk minibatch + daha fazla epoch kullanilir
(--heavy-gpu ile daha da agresif). Dashboard'da GPU kullanim ortalamasi dusuk kalabilir; bu mimari normaldir.

Ornek (5090 + EPYC):
    python train.py --timesteps 50000000 --n-envs 128 --heavy-gpu --compile-policy
    python train.py --timesteps 50000000 --n-envs 128 --vec-env dummy --n-envs 4   # Windows debug
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime

# PERF: Çok sayıda Subproc worker varken her biri OpenMP thread açarsa EPYC thread'leri şişer.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rl.gym_env import BlockBlastGymEnv
from rl.callbacks import TrainingMetricsCallback, HeuristicComparisonCallback
from agents.heuristic_agent import HeuristicAgent


def make_env(seed: int = 0):
    """Monitor ile sarılmış BlockBlastGymEnv üreten factory (tek ortam / eval)."""
    return Monitor(BlockBlastGymEnv(seed=seed))


class EvalEpisodeSeedWrapper(gym.Wrapper):
    """LOG FIX: MaskableEvalCallback + deterministic=True + hep aynı env tohumunda reset,
    aynı 20 episode'u tekrarlar → episode_reward '±0.00', episode_length sabit gibi yanıltıcı çıktı.
    seed=None ile gelen her reset'te artan tohum kullanır (dışarıdan seed verilirse aynen iletir).
    """

    def __init__(self, env: gym.Env, base_seed: int = 9999) -> None:
        super().__init__(env)
        self._base_seed = base_seed
        self._episode_idx = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple:
        if seed is not None:
            return self.env.reset(seed=seed, options=options)
        eff = self._base_seed + self._episode_idx
        self._episode_idx += 1
        return self.env.reset(seed=eff, options=options)


def parse_policy_arch(s: str) -> list[int]:
    """CLI: --policy-net-arch 512,512,256"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--policy-net-arch boş olamaz")
    out: list[int] = []
    for p in parts:
        try:
            v = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Geçersiz katman boyutu: {p}") from exc
        if v < 1:
            raise argparse.ArgumentTypeError(f"Katman boyutu >= 1 olmalı: {v}")
        out.append(v)
    return out


def linear_schedule(initial_value: float, final_value: float):
    """Eğitim ilerledikçe `initial_value`'dan `final_value`'ya doğru lineer azalan schedule.

    SB3 callable schedule API'si: progress_remaining 1.0 (başlangıç) -> 0.0 (bitiş).
    """

    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func


class SaveVecNormalizeCallback(BaseCallback):
    """Yeni bir 'best' model bulunduğunda VecNormalize istatistiklerini de senkron kaydeder.

    MaskableEvalCallback'in `callback_on_new_best` parametresine geçilir; eval esnasında
    yalnızca yeni rekor kırıldığında çağrılır.
    """

    def __init__(self, save_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            vec_norm.save(self.save_path)
            if self.verbose > 0:
                print(f"  -> Yeni en iyi VecNormalize kaydedildi: {self.save_path}")
        return True


class TeeLogger:
    """stdout/stderr'i hem terminale hem log dosyasina yazar.

    Kullanim:
        tee = TeeLogger("logs/training_20250511.log")
        ...egitim...
        tee.close()   # dosya kapanir, orijinal stdout geri gelir
    """

    def __init__(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self._file = open(filepath, "a", encoding="utf-8", buffering=1)  # line-buffered
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self  # type: ignore[assignment]
        sys.stderr = self  # type: ignore[assignment]
        self.filepath = filepath

    def write(self, data: str) -> int:
        if data:
            self._orig_stdout.write(data)
            try:
                self._file.write(data)
            except Exception:
                pass
        return len(data) if data else 0

    def flush(self) -> None:
        self._orig_stdout.flush()
        try:
            self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        try:
            self._file.close()
        except Exception:
            pass

    # encoding/fileno for compatibility
    @property
    def encoding(self) -> str:
        return getattr(self._orig_stdout, "encoding", "utf-8")

    def fileno(self) -> int:
        return self._orig_stdout.fileno()

    def isatty(self) -> bool:
        return False


def train(args: argparse.Namespace) -> None:
    """MaskablePPO modelini egit ve kaydet."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/ppo_v{args.version}_{timestamp}/"

    # ---- LOG FILE: Tum ciktiyi otomatik olarak dosyaya yaz ----
    log_txt_path = f"./logs/training_v{args.version}_{timestamp}.log"
    tee = TeeLogger(log_txt_path)
    print(f"[LOG] Tum cikti su dosyaya kaydediliyor: {log_txt_path}")

    # ---- GPU DIAGNOSTIK ----
    print(f"\n{'='*60}")
    print(f"GPU / CUDA DIAGNOSTIK")
    print(f"{'='*60}")
    print(f"  PyTorch version  : {torch.__version__}")
    print(f"  CUDA available   : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        print(f"  CUDA device      : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version     : {torch.version.cuda}")
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)
        print(f"  VRAM total       : {vram_total:.1f} GB")
        print(f"  VRAM free        : {vram_free:.1f} GB")
        print(f"  cuDNN benchmark  : True")
        print(f"  TF32 matmul      : high")
        print(f"  --> MODEL GPU'DA CALISACAK")
    else:
        device = "cpu"
        print(f"  --> !!! CUDA BULUNAMADI !!!")
        print(f"  --> Model CPU'da calisacak (YAVAS!)")
        print(f"  --> Vast.ai icin: pip install torch --index-url https://download.pytorch.org/whl/cu124")
    print(f"{'='*60}\n")

    best_model_dir = f"models/ppo_v{args.version}_best"
    best_vecnorm_path = os.path.join(best_model_dir, "vecnormalize.pkl")
    os.makedirs(best_model_dir, exist_ok=True)

    # GPU-Vectorized Environment: tüm ortamlar tek GPU tensöründe çalışır.
    use_gpu_env = args.gpu_env and device == "cuda"
    if use_gpu_env:
        from rl.gpu_vec_env import GpuVecEnv
        print(f"\nGPU ENV: {args.n_envs} paralel ortam tek GPU tensöründe çalışacak!")
        print(f"         (SubprocVecEnv yerine GpuVecEnv — ~10x hızlanma)")
        train_env = GpuVecEnv(n_envs=args.n_envs, device=torch.device("cuda"))
        train_env = VecNormalize(
            train_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
        )
    else:
        # PERF: DummyVecEnv = ortamlar tek çekirdekte sırayla → GPU aç kalır.
        # SubprocVecEnv = her ortam ayrı işlem → EPYC çekirdekleri doldurur, rollout toplu gelir, GPU daha sık beslenir.
        if args.vec_env == "dummy" or args.n_envs == 1:
            vec_cls = DummyVecEnv
        else:
            vec_cls = SubprocVecEnv

        train_env = make_vec_env(
            BlockBlastGymEnv,
            n_envs=args.n_envs,
            seed=0,
            wrapper_class=Monitor,
            vec_env_cls=vec_cls,
        )
        train_env = VecNormalize(
            train_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
        )

    heuristic_eval_env = Monitor(BlockBlastGymEnv(seed=9999))

    # LOG FIX: Eval tek tohum + deterministic → ±0.00; rollout ile kıyaslanamaz.
    _eval_base = args.eval_base_seed
    eval_env = DummyVecEnv(
        [lambda: EvalEpisodeSeedWrapper(Monitor(BlockBlastGymEnv(seed=None)), base_seed=_eval_base)]
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=False,
        training=False,
    )

    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError(
            f"batch_size ({args.batch_size}) rollout'tan büyük olamaz: "
            f"n_steps * n_envs = {rollout_size}"
        )

    # PERF / GPU: Ortam adımları CPU'da — GPU yalnızca PPO update sırasında yüklenir.
    # [256,256] gibi küçük MLP'de kernel süresi µs mertebesinde → dashboard'da %1 GPU normal sayılır.
    # CUDA'da otomatik: daha geniş MLP + daha büyük minibatch + daha çok epoch = uzun GPU fazı + daha çok VRAM.
    policy_arch_pi: list[int]
    policy_arch_vf: list[int]
    eff_batch = args.batch_size
    eff_epochs = args.n_epochs

    if args.policy_net_arch:
        policy_arch_pi = list(args.policy_net_arch)
        policy_arch_vf = list(args.policy_net_arch)
    elif device == "cuda" and not args.light_gpu:
        if args.heavy_gpu:
            policy_arch_pi = [1024, 1024, 512]
            policy_arch_vf = [1024, 1024, 512]
            floor_batch = min(65536, rollout_size)
            epoch_floor = 10
        else:
            policy_arch_pi = [512, 512, 256]
            policy_arch_vf = [512, 512, 256]
            floor_batch = min(32768, rollout_size)
            epoch_floor = 8
        if args.no_gpu_auto_tune:
            eff_batch = min(args.batch_size, rollout_size)
            eff_epochs = args.n_epochs
        else:
            eff_batch = max(args.batch_size, floor_batch)
            eff_batch = min(eff_batch, rollout_size)
            eff_epochs = max(args.n_epochs, epoch_floor)
        if not args.no_gpu_auto_tune and (
            eff_batch != args.batch_size or eff_epochs != args.n_epochs
        ):
            print(
                f"PERF GPU: batch_size {args.batch_size}→{eff_batch}, "
                f"n_epochs {args.n_epochs}→{eff_epochs} "
                f"(CUDA otomatik; --light-gpu küçük ağ, --no-gpu-auto-tune sadece batch/epoch artışını kapatır)"
            )
    else:
        policy_arch_pi = [256, 256]
        policy_arch_vf = [256, 256]

    policy_kwargs = dict(net_arch=[dict(pi=policy_arch_pi, vf=policy_arch_vf)])

    if args.resume:
        model = MaskablePPO.load(args.resume, env=train_env, device=device)
        model.tensorboard_log = log_dir
        print(f"Model yüklendi: {args.resume}")
    else:
        model = MaskablePPO(
            "MlpPolicy",
            env=train_env,
            # GPU UPDATE / mimari: LR artık sabit (linear decay kaldırıldı — son steplerde donmayı önlemek için).
            learning_rate=3e-4,
            # PERF: Büyük rollout (n_steps * n_envs) GPU'da daha verimli minibatch işler.
            n_steps=args.n_steps,
            batch_size=eff_batch,
            n_epochs=eff_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=linear_schedule(0.2, 0.1),
            # Entropy bonus: düşük ent_coef → politika çabuk deterministikleşir (entropy 2.5→1.5 gibi).
            # Maskeli ayrık aksiyonda keşif için 0.1–0.2 aralığı sık denenir; CLI ile ayarla.
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.02,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
        )

        if args.compile_policy and device == "cuda" and hasattr(torch, "compile"):
            try:
                model.policy = torch.compile(model.policy, mode="reduce-overhead")  # type: ignore[assignment]
                print("PERF GPU: torch.compile(policy) etkin.")
            except Exception as exc:
                print(f"PERF GPU: torch.compile atlandı ({exc})")

    metrics_cb = TrainingMetricsCallback(log_freq=1000)
    heuristic_cb = HeuristicComparisonCallback(
        eval_env=heuristic_eval_env,
        heuristic_agent=HeuristicAgent(),
        eval_freq=20_000,
        n_eval_episodes=20,
    )

    save_vecnorm_cb = SaveVecNormalizeCallback(
        save_path=best_vecnorm_path,
        verbose=1,
    )
    eval_cb = MaskableEvalCallback(
        eval_env=eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        use_masking=True,
        callback_on_new_best=save_vecnorm_cb,
        verbose=1,
    )

    print(f"\n{'='*50}")
    print(f"Block Blast MaskablePPO Eğitimi Başlıyor")
    print(f"Toplam adım  : {args.timesteps:,}")
    print(f"Paralel ortam: {args.n_envs} ({args.vec_env})")
    print(f"Rollout/adım : n_steps={args.n_steps} → buffer={rollout_size:,} geçiş/PPO iter")
    eff_b = eff_batch if not args.resume else getattr(model, "batch_size", args.batch_size)
    eff_e = eff_epochs if not args.resume else getattr(model, "n_epochs", args.n_epochs)
    print(f"batch_size   : {eff_b}  |  n_epochs: {eff_e}")
    if not args.resume:
        print(f"policy net   : pi={policy_arch_pi}, vf={policy_arch_vf}")
    else:
        print("policy net   : (checkpoint — yukarıdaki mimari yalnızca sıfırdan eğitim için geçerli)")
    print(f"ent_coef     : {args.ent_coef}")
    print(f"eval tohum   : base={args.eval_base_seed} (her eval episode +1, LOG FIX)")
    print(f"Cihaz        : {device}")
    print(f"Model versiyonu: v{args.version}")
    print(f"Log dizini   : {log_dir}")
    print(f"{'='*50}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=[metrics_cb, heuristic_cb, eval_cb],
        reset_num_timesteps=not bool(args.resume),
    )

    last_model_path = f"models/ppo_v{args.version}_last"
    last_vecnorm_path = f"models/ppo_v{args.version}_last_vecnorm.pkl"
    model.save(last_model_path)
    train_env.save(last_vecnorm_path)

    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    print(f"\nSon model         : {last_model_path}.zip")
    print(f"Son VecNormalize  : {last_vecnorm_path}")
    print(f"En iyi model      : {best_model_path}")
    print(f"En iyi VecNormalize: {best_vecnorm_path}")
    print(f"TensorBoard loglari: {log_dir}")
    print(f"TensorBoard baslatmak icin: tensorboard --logdir ./logs/ --port 6006")
    print(f"Egitim logu       : {log_txt_path}")

    # Log dosyasini kapat
    tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Block Blast MaskablePPO Eğitimi")
    # GPU UPDATE: varsayılan en az 2M adım (uzun PPO koşusu).
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Toplam eğitim adımı (örn. 50_000_000)")
    parser.add_argument("--resume", type=str, default=None, help="Devam edilecek model yolu")
    parser.add_argument("--version", type=int, default=3, help="Model versiyonu")
    parser.add_argument(
        "--n-envs",
        type=int,
        default=64,
        help="Paralel ortam sayısı (EPYC+5090 için 64–128 deneyin; thread şişmesini OMP_NUM_THREADS=1 engeller)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Ortam başına rollout uzunluğu; buffer = n_steps * n_envs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="PPO minibatch (<= n_steps * n_envs); CUDA varsayilaninda otomatik yukseltilebilir.",
    )
    parser.add_argument(
        "--vec-env",
        type=str,
        choices=("subproc", "dummy"),
        default="subproc",
        help="subproc=çok işlem (Linux/Vast önerilir), dummy=tek işlem",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.12,
        help="Entropy katsayısı; düşük değer hızlı entropy çöküşü (daha deterministik politika). "
        "Çok paralel ortamda hâlâ çöküyorsa 0.15–0.25 veya --n-epochs 2 dene.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=4,
        help="PPO epoch sayısı; büyük rollout'ta 2–3 keşfi korur, 4+ daha agresif günceller.",
    )
    parser.add_argument(
        "--eval-base-seed",
        type=int,
        default=9999,
        help="MaskableEvalCallback ortamı: her episode farklı tahta için başlangıç tohumu (9999, 10000, …).",
    )
    parser.add_argument(
        "--light-gpu",
        action="store_true",
        help="Küçük MLP [256,256] ve CUDA batch/epoch otomatiğini kullanma (eski hızlı deneme profili).",
    )
    parser.add_argument(
        "--heavy-gpu",
        action="store_true",
        help="CUDA'da daha geniş MLP [1024,1024,512] ve daha agresif batch/epoch tabanı (5090 için).",
    )
    parser.add_argument(
        "--no-gpu-auto-tune",
        action="store_true",
        help="CUDA'da bile batch_size / n_epochs için otomatik tabanı uygulama (policy genişliği --heavy-gpu ile kalır).",
    )
    parser.add_argument(
        "--policy-net-arch",
        type=parse_policy_arch,
        default=None,
        metavar="SIZES",
        help="Özel pi/vf MLP: virgüllü liste, örn: 512,512,256",
    )
    parser.add_argument(
        "--compile-policy",
        action="store_true",
        help="CUDA'da torch.compile(policy) dene (PyTorch 2.x; ilk iterasyon yavaş olabilir).",
    )
    parser.add_argument(
        "--gpu-env",
        action="store_true",
        help="GPU-vectorized ortam kullan (CUDA gerekli). "
        "Tüm ortamlar tek GPU tensöründe çalışır — SubprocVecEnv yerine. "
        "RTX 5090 / Vast.ai için önerilir: --gpu-env --n-envs 2048",
    )
    args = parser.parse_args()
    if args.n_envs < 1:
        parser.error("--n-envs en az 1 olmalı")
    if args.n_epochs < 1:
        parser.error("--n-epochs en az 1 olmalı")
    if args.ent_coef < 0:
        parser.error("--ent-coef negatif olamaz")
    if args.light_gpu and args.heavy_gpu:
        parser.error("--light-gpu ve --heavy-gpu birlikte kullanılamaz")
    if args.vec_env == "subproc" and args.n_envs > 1 and sys.platform == "win32":
        print(
            "NOT: Windows'ta SubprocVecEnv bazen spawn/pickle sorunu çıkarır; "
            "sorun olursa --vec-env dummy --n-envs 4 deneyin."
        )
    train(args)
