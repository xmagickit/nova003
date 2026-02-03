import os
from traceback import print_exc
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import json
import time
import subprocess
import argparse
import bittensor as bt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import pandas as pd
from pathlib import Path
import nova_ph2
from itertools import combinations

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PARENT_DIR)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
SHARED_DIR = os.path.join(OUTPUT_DIR, "shared")

from nova_ph2.PSICHIC.wrapper import PsichicWrapper
from nova_ph2.PSICHIC.psichic_utils.data_utils import virtual_screening
from molecules import (
    generate_valid_random_molecules_batch,
    select_diverse_elites,
    build_component_weights,
    compute_tanimoto_similarity_to_pool,
    sample_random_valid_molecules,
    compute_maccs_entropy,
    SynthonLibrary,
    generate_molecules_from_synthon_library,
    validate_molecules,
    generate_inchikey,
)

DB_PATH = str(Path(nova_ph2.__file__).resolve().parent / "combinatorial_db" / "molecules.sqlite")


# PHASE 2: Reaction-specific configurations for optimal performance
REACTION_CONFIGS = {
    1: {
        'base_samples': 1200,       # 2-comp, fast reaction
        'elite_frac': 0.70,
        'mutation_prob': 0.20,
        'synthon_ratio': 0.75,
        'exploit_start_iter': 3,
        'first_iter_mult': 6
    },
    2: {
        'base_samples': 1100,       # 2-comp, moderate
        'elite_frac': 0.75,
        'mutation_prob': 0.15,
        'synthon_ratio': 0.78,
        'exploit_start_iter': 3,
        'first_iter_mult': 6
    },
    3: {
        'base_samples': 800,        # 3-comp, slower
        'elite_frac': 0.65,
        'mutation_prob': 0.25,
        'synthon_ratio': 0.70,
        'exploit_start_iter': 4,
        'first_iter_mult': 5
    },
    4: {
        'base_samples': 1150,       # 2-comp
        'elite_frac': 0.72,
        'mutation_prob': 0.18,
        'synthon_ratio': 0.76,
        'exploit_start_iter': 3,
        'first_iter_mult': 6
    },
    5: {
        'base_samples': 750,        # 3-comp, slowest
        'elite_frac': 0.68,
        'mutation_prob': 0.22,
        'synthon_ratio': 0.68,
        'exploit_start_iter': 4,
        'first_iter_mult': 4
    },
    6: {
        'base_samples': 1200,       # 2-comp, fast
        'elite_frac': 0.75,
        'mutation_prob': 0.15,
        'synthon_ratio': 0.78,
        'exploit_start_iter': 3,
        'first_iter_mult': 6
    }
}


target_models = []
antitarget_models = []
exploit_worker_process = None

# PHASE 3: Memory Management - DataFrame Pool
class DataFramePool:
    """PHASE 3: Object pool for DataFrame reuse to reduce allocations."""
    def __init__(self, pool_size=10):
        self.pool = []
        self.pool_size = pool_size
        
    def acquire(self):
        """Get a DataFrame from pool or create new one."""
        if self.pool:
            df = self.pool.pop()
            return df
        return pd.DataFrame()
    
    def release(self, df):
        """Return DataFrame to pool after clearing it."""
        if len(self.pool) < self.pool_size:
            try:
                df.drop(df.index, inplace=True)  # Clear data
                self.pool.append(df)
            except:
                pass  # If error, just discard

# Global DataFrame pool
_dataframe_pool = DataFramePool(pool_size=20)


def load_warm_start_molecules(rxn_id: int) -> pd.DataFrame:
    """
    PHASE 3: Load previously successful molecules for this reaction as warm start.
    """
    warm_start_file = os.path.join(OUTPUT_DIR, f"warm_start_rxn{rxn_id}.json")
    try:
        if os.path.exists(warm_start_file):
            with open(warm_start_file) as f:
                data = json.load(f)
            if 'molecules' in data and data['molecules']:
                df = pd.DataFrame(data['molecules'])
                bt.logging.info(f"[WarmStart] Loaded {len(df)} molecules from previous run for rxn:{rxn_id}")
                return df
    except Exception as e:
        bt.logging.warning(f"[WarmStart] Could not load warm start data: {e}")
    return pd.DataFrame()


def save_warm_start_molecules(rxn_id: int, top_molecules: pd.DataFrame):
    """
    PHASE 3: Save top molecules for future runs as warm start.
    """
    if top_molecules.empty:
        return
    
    warm_start_file = os.path.join(OUTPUT_DIR, f"warm_start_rxn{rxn_id}.json")
    try:
        top_20 = top_molecules.head(20)[["name", "smiles", "InChIKey", "score"]].copy()
        data = {
            'molecules': top_20.to_dict('records'),
            'timestamp': time.time(),
            'rxn_id': rxn_id
        }
        with open(warm_start_file, 'w') as f:
            json.dump(data, f, indent=2)
        bt.logging.info(f"[WarmStart] Saved {len(top_20)} molecules for future runs")
    except Exception as e:
        bt.logging.warning(f"[WarmStart] Could not save warm start data: {e}")


def spawn_exploit_worker():
    """Spawn the exploit worker as a subprocess."""
    global exploit_worker_process
    try:
        os.makedirs(SHARED_DIR, exist_ok=True)
        bt.logging.info("[Miner] Spawning exploit_worker.py subprocess...")
        exploit_worker_path = os.path.join(BASE_DIR, "exploit_worker.py")
        log_path = os.path.join(OUTPUT_DIR, "exploit_worker.log")

        with open(log_path, "w") as log_file:
            exploit_worker_process = subprocess.Popen(
                [sys.executable, exploit_worker_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env={**os.environ, "OUTPUT_DIR": OUTPUT_DIR},
                cwd=BASE_DIR
            )
        bt.logging.info(f"[Miner] Exploit worker started with PID {exploit_worker_process.pid}")
    except Exception as e:
        bt.logging.warning(f"[Miner] Failed to spawn exploit worker: {e}")
        exploit_worker_process = None


def stop_exploit_worker():
    """Stop the exploit worker subprocess."""
    global exploit_worker_process
    if exploit_worker_process is not None:
        try:
            exploit_worker_process.terminate()
            exploit_worker_process.wait(timeout=5)
            bt.logging.info("[Miner] Exploit worker terminated")
        except Exception as e:
            bt.logging.warning(f"[Miner] Error stopping exploit worker: {e}")
            try:
                exploit_worker_process.kill()
            except:
                pass


def write_top10_for_exploit_worker(top_pool: pd.DataFrame):
    """Write top 100 molecules to shared file for exploit worker."""
    try:
        os.makedirs(SHARED_DIR, exist_ok=True)
        top10_path = os.path.join(SHARED_DIR, "top10.json")
        top10 = top_pool.head(100)[["name", "smiles", "score"]].copy()
        top10_data = {
            "molecules": top10.to_dict("records"),
            "timestamp": time.time()
        }
        tmp_path = top10_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(top10_data, f)
        os.rename(tmp_path, top10_path)
    except Exception as e:
        bt.logging.warning(f"[Miner] Failed to write top10.json: {e}")


def read_pool_b() -> pd.DataFrame:
    """Read Pool B results from exploit worker."""
    try:
        pool_b_path = os.path.join(SHARED_DIR, "pool_b.json")
        if not os.path.exists(pool_b_path):
            return pd.DataFrame()
        with open(pool_b_path) as f:
            data = json.load(f)
        molecules = data.get("molecules", [])
        if not molecules:
            return pd.DataFrame()
        df = pd.DataFrame(molecules)
        bt.logging.info(f"[Miner] Read {len(df)} molecules from Pool B "
                       f"(avg={data.get('avg_score', 0):.4f}, max={data.get('max_score', 0):.4f})")
        return df
    except Exception as e:
        bt.logging.warning(f"[Miner] Failed to read pool_b.json: {e}")
        return pd.DataFrame()


def merge_pools(pool_a: pd.DataFrame, pool_b: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Merge Pool A and Pool B, deduplicate by InChIKey, keep top num_molecules."""
    if pool_b.empty:
        bt.logging.info("[Miner] Pool B empty, using Pool A only")
        return pool_a.head(config["num_molecules"])

    bt.logging.info(f"[Miner] Merging Pool A ({len(pool_a)}) + Pool B ({len(pool_b)})")

    if "InChIKey" not in pool_b.columns:
        pool_b = pool_b.copy()
        pool_b["InChIKey"] = pool_b["smiles"].apply(generate_inchikey)

    combined = pd.concat([pool_a, pool_b], ignore_index=True)
    combined = combined.sort_values(by="score", ascending=False)
    combined = combined.drop_duplicates(subset=["InChIKey"], keep="first")
    bt.logging.info(f"[Miner] Combined pool: {len(combined)} unique molecules")
    return combined.head(config["num_molecules"])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Nova Blueprint Miner")
    parser.add_argument('--rxn', type=int, help='Override allowed_reaction (e.g., --rxn 5)')
    parser.add_argument('--input', type=str, default=os.path.join(BASE_DIR, "input.json"),
                        help='Path to input.json config file')
    return parser.parse_args()


def get_config(input_file: str = os.path.join(BASE_DIR, "input.json"), rxn_override: int = None):
    with open(input_file, "r") as f:
        d = json.load(f)
    config = {**d.get("config", {}), **d.get("challenge", {})}

    # Override allowed_reaction if --rxn argument provided
    if rxn_override is not None:
        config["allowed_reaction"] = f"rxn:{rxn_override}"
        bt.logging.warning(f"[Miner] Overriding allowed_reaction to rxn:{rxn_override}")

    return config


def initialize_models(config: dict):
    """Initialize separate model instances for each target and antitarget sequence."""
    global target_models, antitarget_models
    target_models = []
    antitarget_models = []

    for seq in config["target_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        target_models.append(wrapper)

    for seq in config["antitarget_sequences"]:
        wrapper = PsichicWrapper()
        wrapper.initialize_model(seq)
        antitarget_models.append(wrapper)


# ---------- scoring helpers (reuse pre-initialized models) ----------
def target_score_from_data(data: pd.Series):
    """Score molecules against all target models."""
    global target_models, antitarget_models
    try:
        target_scores = []
        smiles_list = data.tolist()
        for target_model in target_models:
            scores = target_model.score_molecules(smiles_list)
            for antitarget_model in antitarget_models:
                antitarget_model.smiles_list = smiles_list
                antitarget_model.smiles_dict = target_model.smiles_dict

            scores.rename(columns={'predicted_binding_affinity': "target"}, inplace=True)
            target_scores.append(scores["target"])
        # Average across all targets
        target_series = pd.DataFrame(target_scores).mean(axis=0)
        return target_series
    except Exception as e:
        bt.logging.error(f"Target scoring error: {e}")
        return pd.Series(dtype=float)


def antitarget_scores():
    """Score molecules against all antitarget models."""

    global antitarget_models
    try:
        antitarget_scores = []
        for i, antitarget_model in enumerate(antitarget_models):
            antitarget_model.create_screen_loader(antitarget_model.protein_dict, antitarget_model.smiles_dict)
            antitarget_model.screen_df = virtual_screening(antitarget_model.screen_df,
                                            antitarget_model.model,
                                            antitarget_model.screen_loader,
                                            os.getcwd(),
                                            save_interpret=False,
                                            ligand_dict=antitarget_model.smiles_dict,
                                            device=antitarget_model.device,
                                            save_cluster=False,
                                            )
            scores = antitarget_model.screen_df[['predicted_binding_affinity']]
            scores.rename(columns={'predicted_binding_affinity': f"anti_{i}"}, inplace=True)
            antitarget_scores.append(scores[f"anti_{i}"])

        if not antitarget_scores:
            return pd.Series(dtype=float)

        # average across antitargets
        anti_series = pd.DataFrame(antitarget_scores).mean(axis=0)
        return anti_series
    except Exception as e:
        bt.logging.error(f"Antitarget scoring error: {e}")
        return pd.Series(dtype=float)


def _cpu_random_candidates_with_similarity(
    iteration: int,
    n_samples: int,
    subnet_config: dict,
    top_pool_df: pd.DataFrame,
    avoid_inchikeys: set[str] | None = None,
    thresh: float = 0.8
) -> pd.DataFrame:
    """
    CPU-side helper:
    - draws a random batch of valid molecules (independent of the GPU batch),
    - computes Tanimoto similarity vs. current top_pool,
    - returns a DataFrame with name, smiles, InChIKey, tanimoto_similarity.
    """
    try:
        random_df = sample_random_valid_molecules(
            n_samples=n_samples,
            subnet_config=subnet_config,
            avoid_inchikeys=avoid_inchikeys,
            focus_neighborhood_of=top_pool_df
        )
        if random_df.empty or top_pool_df.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

        sims = compute_tanimoto_similarity_to_pool(
            candidate_smiles=random_df["smiles"],
            pool_smiles=top_pool_df["smiles"],
        )
        random_df = random_df.copy()
        random_df["tanimoto_similarity"] = sims.reindex(random_df.index).fillna(0.0)
        random_df = random_df.sort_values(by="tanimoto_similarity", ascending=False)
        random_df_filtered = random_df[random_df["tanimoto_similarity"] >= thresh]

        if random_df_filtered.empty:
            return pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])

        random_df_filtered = random_df_filtered.reset_index(drop=True)
        return random_df_filtered[["name", "smiles", "InChIKey"]]
    except Exception as e:
        bt.logging.warning(f"[Miner] _cpu_random_candidates_with_similarity failed: {e}")
        return pd.DataFrame(columns=["name", "smiles", "InChIKey"])

def select_diverse_subset(pool, top_95_smiles, subset_size=5, entropy_threshold=0.1):
    smiles_list = pool["smiles"].tolist()
    for combination in combinations(smiles_list, subset_size):
        test_subset = top_95_smiles + list(combination)
        entropy = compute_maccs_entropy(test_subset)
        if entropy >= entropy_threshold:
            bt.logging.info(f"Entropy Threshold Met: {entropy:.4f}")
            return pool[pool["smiles"].isin(combination)]

    bt.logging.warning("No combination exceeded the given entropy threshold.")
    return pd.DataFrame()


def dynamic_batch_size(iteration: int, time_remaining: float, current_score: float, base_batch: int = 400) -> int:
    """
    PHASE 2: Dynamic batch sizing based on time, score, and iteration.
    Returns optimal batch size for current situation.
    """
    # Base multiplier from time remaining
    if time_remaining > 1500:      # 25분 이상
        time_mult = 1.0
    elif time_remaining > 900:     # 15분 이상
        time_mult = 0.95
    elif time_remaining > 600:     # 10분 이상
        time_mult = 0.90
    else:                          # 10분 미만
        time_mult = 0.85
    
    # Score-based adjustment
    if current_score > 0.015:
        score_mult = 0.9  # High score: smaller batches for focused exploitation
    elif current_score > 0.010:
        score_mult = 1.0  # Medium score: normal batch size
    else:
        score_mult = 1.1  # Low score: larger batches for more exploration
    
    # Iteration-based (early iterations need more exploration)
    if iteration <= 3:
        iter_mult = 1.2
    elif iteration <= 10:
        iter_mult = 1.0
    else:
        iter_mult = 0.95
    
    batch_size = int(base_batch * time_mult * score_mult * iter_mult)
    return max(300, min(600, batch_size))  # Clamp between 300-600


def adaptive_strategy(top_pool: pd.DataFrame, iteration: int, time_remaining: float, rxn_config: dict) -> dict:
    """
    PHASE 2: Adaptive strategy based on current scores and situation.
    Returns strategy parameters to use for this iteration.
    """
    if top_pool.empty:
        return {
            'mode': 'explore',
            'n_samples_mult': 1.2,
            'similarity': 0.55,
            'n_per_base': 30,
            'elite_frac': 0.60
        }
    
    current_max = top_pool['score'].max()
    current_avg = top_pool['score'].mean()
    
    # Very High Score: Ultra-tight exploitation
    if current_max > 0.018:
        bt.logging.info(f"[Adaptive] Ultra-exploit mode (max={current_max:.4f})")
        return {
            'mode': 'ultra_exploit',
            'n_samples_mult': 0.8,
            'similarity': 0.93,
            'n_per_base': 120,
            'elite_frac': 0.85
        }
    
    # High Score: Tight exploitation
    elif current_max > 0.012:
        bt.logging.info(f"[Adaptive] Tight-exploit mode (max={current_max:.4f})")
        return {
            'mode': 'tight_exploit',
            'n_samples_mult': 0.9,
            'similarity': 0.85,
            'n_per_base': 80,
            'elite_frac': 0.78
        }
    
    # Medium Score: Balanced
    elif current_max > 0.008:
        bt.logging.info(f"[Adaptive] Balanced mode (max={current_max:.4f})")
        return {
            'mode': 'balanced',
            'n_samples_mult': 1.0,
            'similarity': 0.70,
            'n_per_base': 50,
            'elite_frac': 0.72
        }
    
    # Low Score: Explore more
    else:
        bt.logging.info(f"[Adaptive] Explore mode (max={current_max:.4f})")
        return {
            'mode': 'explore',
            'n_samples_mult': 1.2,
            'similarity': 0.55,
            'n_per_base': 30,
            'elite_frac': rxn_config.get('elite_frac', 0.60)
        }


def main(config: dict):
    # WINNING MODEL: Combines best of richard1220v3 (balanced) + patarcom1 (exploration) + smart adaptations
    # Target: Beat richard1220v3 (0.0124) by 0.05+ to reach 0.0174+
    print(f"[Miner] main() started, rxn={config['allowed_reaction']}", flush=True)
    
    # PHASE 2: Get reaction-specific configuration
    rxn_id = int(config["allowed_reaction"].split(":")[-1])
    rxn_config = REACTION_CONFIGS.get(rxn_id, REACTION_CONFIGS[1])  # Default to rxn:1 if not found
    
    base_n_samples = rxn_config['base_samples']  # PHASE 2: Reaction-specific
    mutation_prob = rxn_config['mutation_prob']  # PHASE 2: Reaction-specific
    elite_frac = rxn_config['elite_frac']  # PHASE 2: Reaction-specific
    exploit_start_iter = rxn_config['exploit_start_iter']  # PHASE 2: Reaction-specific
    
    print(f"[Miner] rxn_id={rxn_id}, base_samples={base_n_samples}, elite_frac={elite_frac}, mutation={mutation_prob} [PHASE 2]", flush=True)
    
    top_pool = pd.DataFrame(columns=["name", "smiles", "InChIKey", "score", "Target", "Anti"])
    iteration = 0

    # PHASE 3: Load warm start molecules
    warm_start_df = load_warm_start_molecules(rxn_id)
    if not warm_start_df.empty:
        # Add warm start molecules to initial pool
        top_pool = pd.concat([top_pool, warm_start_df], ignore_index=True)
        bt.logging.info(f"[PHASE 3] Warm start: {len(warm_start_df)} molecules loaded")

    seen_inchikeys = set()
    seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
    start = time.time()
    prev_avg_score = None
    current_avg_score = None
    score_improvement_rate = 0.0
    no_improvement_counter = 0

    synthon_lib = None
    use_synthon_search = False

    # Track best molecules and score history for adaptive strategies
    best_molecules_history = []
    max_score_history = []

    # PHASE 2: Reaction-specific first iteration boost
    n_samples_first_iteration = base_n_samples * rxn_config['first_iter_mult']

    # PHASE 2: Reaction-specific exploit worker spawn time
    exploit_worker_spawned = False
    print(f"[Miner] Entering main loop (exploit worker will start after iteration {exploit_start_iter}) [PHASE 2]", flush=True)

    # PHASE 1 OPTIMIZATION: Increased CPU workers from 2 to 4 for better parallelization
    with ProcessPoolExecutor(max_workers=4) as cpu_executor:
        while time.time() - start < 1800:
            iteration += 1
            iter_start_time = time.time()
            print(f"[Miner] === Starting iteration {iteration} ===", flush=True)

            # PHASE 2: Reaction-specific exploit worker spawn
            if iteration == exploit_start_iter and not exploit_worker_spawned:
                spawn_exploit_worker()
                exploit_worker_spawned = True
                print(f"[Miner] Exploit worker spawned after iteration {exploit_start_iter} (PHASE 2 - Reaction-specific)", flush=True)

            # PHASE 2: Adaptive strategy based on scores and time
            remaining_time = 1800 - (time.time() - start)
            adaptive_params = adaptive_strategy(top_pool, iteration, remaining_time, rxn_config)
            
            # Apply adaptive parameters
            n_samples_mult = adaptive_params['n_samples_mult']
            adaptive_elite_frac = adaptive_params['elite_frac']
            
            # Time-based adjustment (keep some time pressure)
            if remaining_time > 1500:
                time_mult = 1.0
            elif remaining_time > 900:
                time_mult = 0.95
            elif remaining_time > 600:
                time_mult = 0.90
            elif remaining_time > 300:
                time_mult = 0.85
            else:
                time_mult = 0.80
            
            n_samples = int(base_n_samples * n_samples_mult * time_mult)
            
            # PHASE 2: Dynamic batch size
            current_max_score = top_pool['score'].max() if not top_pool.empty else 0.0
            dynamic_batch = dynamic_batch_size(iteration, remaining_time, current_max_score, base_batch=400)
            
            bt.logging.info(f"[Miner] Adaptive mode: {adaptive_params['mode']}, n_samples={n_samples}, batch_size={dynamic_batch}")

            # Build synthon library after first iteration
            if iteration == 2 and not top_pool.empty and synthon_lib is None:
                try:
                    bt.logging.info("[Miner] Building synthon library from top molecules...")
                    synthon_lib_start = time.time()
                    synthon_lib = SynthonLibrary(DB_PATH, rxn_id)
                    use_synthon_search = True
                    bt.logging.info(f"[Miner] Synthon library ready! Built in {time.time() - synthon_lib_start:.2f}s")
                except Exception as e:
                    bt.logging.warning(f"[Miner] Could not build synthon library: {e}")
                    use_synthon_search = False

            component_weights = build_component_weights(top_pool, rxn_id) if not top_pool.empty else None
            # Enhanced elite pool - PHASE 1 OPTIMIZATION: Increased from 150 to 200
            elite_df = select_diverse_elites(top_pool, min(200, len(top_pool))) if not top_pool.empty else pd.DataFrame()
            elite_names = elite_df["name"].tolist() if not elite_df.empty else None

            # WINNING STRATEGY: Intelligent exploration/exploitation balance
            if iteration == 1:
                print(f"[Miner] Iteration 1: Generating {n_samples_first_iteration} molecules for rxn:{rxn_id}...", flush=True)
                bt.logging.info(f"[Miner] Iteration {iteration}: Initial broad random sampling")
                data = generate_valid_random_molecules_batch(
                    rxn_id,
                    n_samples=n_samples_first_iteration,
                    db_path=DB_PATH,
                    subnet_config=config,
                    batch_size=400,
                    elite_names=None,
                    elite_frac=0.0,
                    mutation_prob=1.0,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=None,
                )

            elif use_synthon_search and iteration > 2 and not top_pool.empty:
                bt.logging.info(f"[Miner] Iteration {iteration}: Smart synthon similarity search")

                # Get current max score for adaptive strategy
                current_max_score = top_pool['score'].max() if not top_pool.empty else None
                current_avg_score = top_pool['score'].mean() if not top_pool.empty else None
                max_score_history.append(current_max_score)
                if len(max_score_history) > 5:
                    max_score_history.pop(0)

                # SMART: Adaptive strategy based on improvement rate AND absolute score
                has_high_score = current_max_score is not None and current_max_score > 0.01
                has_very_high_score = current_max_score is not None and current_max_score > 0.015

                # Time-based strategy
                time_elapsed = time.time() - start
                is_late_stage = time_elapsed > 1200
                is_very_late_stage = time_elapsed > 1500

                if score_improvement_rate > 0.05:
                    # High improvement: tight exploration
                    sim_threshold = 0.75
                    n_per_base = 15
                    n_seeds = 20
                    synthon_ratio = 0.75
                    bt.logging.info(f"[Miner] High improvement ({score_improvement_rate:.4f}), tight similarity (0.75)")

                elif score_improvement_rate > 0.02:
                    # Good improvement: medium-tight exploration
                    sim_threshold = 0.70
                    n_per_base = 18
                    n_seeds = 25
                    synthon_ratio = 0.75
                    bt.logging.info(f"[Miner] Good improvement ({score_improvement_rate:.4f}), medium-tight similarity (0.70)")

                elif score_improvement_rate > 0.005:
                    # Moderate improvement: balanced exploration
                    sim_threshold = 0.65
                    n_per_base = 20
                    n_seeds = 30
                    synthon_ratio = 0.70
                    bt.logging.info(f"[Miner] Moderate improvement ({score_improvement_rate:.4f}), medium similarity (0.65)")

                else:
                    # Low/no improvement - PROVEN MULTI-RANGE STRATEGY (like richard1220v3)
                    bt.logging.info(f"[Miner] Low improvement ({score_improvement_rate:.4f}), using PROVEN MULTI-RANGE strategy")

                    # SMART: Adjust strategy based on absolute score and time
                    if has_very_high_score or is_very_late_stage:
                        # When we have very high scores, add focused exploitation on TOP 1
                        # Part 1: Ultra-tight on TOP 1 molecule - PHASE 1 OPTIMIZATION
                        n_synthon_top1 = int(n_samples * 0.30)  # PHASE 1: Increased from 0.25 to 0.30
                        synthon_top1_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(1),  # TOP 1 ONLY
                            n_synthon_top1,
                            min_similarity=0.92,  # PHASE 1: Increased from 0.91 to 0.92 for even tighter similarity
                            n_per_base=100  # PHASE 1: Increased from 50 to 100 for maximum variations
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_top1_df)} TOP-1 synthon candidates (sim=0.92, n_per_base=100) [PHASE 1]")

                        # Part 2: Ultra-tight on top 5 molecules (10% of synthon budget)
                        n_synthon_tight = int(n_samples * 0.07)  # 10% of 70%
                        synthon_tight_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(5),
                            n_synthon_tight,
                            min_similarity=0.80,  # Tight
                            n_per_base=30
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_tight_df)} TIGHT synthon candidates (sim=0.80)")

                        # Part 3: Medium on molecules 10-40 (30% of synthon budget)
                        n_synthon_medium = int(n_samples * 0.21)  # 30% of 70%
                        seed_medium = top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:]
                        synthon_medium_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            seed_medium,
                            n_synthon_medium,
                            min_similarity=0.55,  # Medium - like richard1220v3
                            n_per_base=15
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_medium_df)} MEDIUM synthon candidates (sim=0.55)")

                        # Part 4: Broad on top 50 (30% of synthon budget)
                        n_synthon_broad = int(n_samples * 0.21)  # 30% of 70%
                        synthon_broad_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(50),
                            n_synthon_broad,
                            min_similarity=0.40,  # Broad - like richard1220v3
                            n_per_base=20
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_broad_df)} BROAD synthon candidates (sim=0.40)")

                        # Combine all synthon approaches
                        synthon_df = pd.concat([synthon_top1_df, synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)
                    else:
                        # Standard: PROVEN multi-range strategy from richard1220v3
                        # Part 1: Ultra-tight on top 5 molecules (40% of synthon budget)
                        n_synthon_tight = int(n_samples * 0.28)  # 40% of 70%
                        synthon_tight_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(5),
                            n_synthon_tight,
                            min_similarity=0.80,  # Very tight!
                            n_per_base=30
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_tight_df)} TIGHT synthon candidates (sim=0.80)")

                        # Part 2: Medium on molecules 10-40 (30% of synthon budget)
                        n_synthon_medium = int(n_samples * 0.21)  # 30% of 70%
                        seed_medium = top_pool.iloc[10:40] if len(top_pool) > 40 else top_pool.iloc[5:]
                        synthon_medium_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            seed_medium,
                            n_synthon_medium,
                            min_similarity=0.55,  # Medium - like richard1220v3
                            n_per_base=15
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_medium_df)} MEDIUM synthon candidates (sim=0.55)")

                        # Part 3: Broad on top 50 (30% of synthon budget)
                        n_synthon_broad = int(n_samples * 0.21)  # 30% of 70%
                        synthon_broad_df = generate_molecules_from_synthon_library(
                            synthon_lib,
                            top_pool.head(50),
                            n_synthon_broad,
                            min_similarity=0.40,  # Broad - like richard1220v3
                            n_per_base=20
                        )
                        bt.logging.info(f"[Miner] Generated {len(synthon_broad_df)} BROAD synthon candidates (sim=0.40)")

                        # Combine all synthon approaches
                        synthon_df = pd.concat([synthon_tight_df, synthon_medium_df, synthon_broad_df], ignore_index=True)

                    synthon_df = synthon_df.drop_duplicates(subset=["name"], keep="first")

                    if not synthon_df.empty:
                        synthon_df = validate_molecules(synthon_df, config)
                        bt.logging.info(f"[Miner] {len(synthon_df)} multi-range synthon candidates passed validation")

                    # Generate remaining from GA with component weighting
                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_traditional,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=dynamic_batch,  # PHASE 2: Dynamic batch size
                            elite_names=elite_names,
                            elite_frac=adaptive_elite_frac,  # PHASE 2: Adaptive elite fraction
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[Miner] Combined: {len(data)} total ({len(synthon_df)} multi-range synthon + {len(traditional_df)} GA)")

                    # Skip the standard synthon generation below
                    synthon_df = None

                # Standard single-range synthon generation (for high/medium improvement)
                if score_improvement_rate > 0.005:  # Only if not using multi-range
                    n_synthon = int(n_samples * synthon_ratio)
                    synthon_gen_start = time.time()
                    synthon_df = generate_molecules_from_synthon_library(
                        synthon_lib,
                        top_pool.head(n_seeds),
                        n_synthon,
                        min_similarity=sim_threshold,
                        n_per_base=n_per_base
                    )
                    bt.logging.info(f"[Miner] Generated {len(synthon_df)} synthon candidates in {time.time() - synthon_gen_start:.2f}s")

                    # Generate remaining from traditional method
                    n_traditional = n_samples - len(synthon_df)
                    if n_traditional > 0:
                        traditional_df = generate_valid_random_molecules_batch(
                            rxn_id,
                            n_samples=n_traditional,
                            db_path=DB_PATH,
                            subnet_config=config,
                            batch_size=dynamic_batch,  # PHASE 2: Dynamic batch size
                            elite_names=elite_names,
                            elite_frac=adaptive_elite_frac,  # PHASE 2: Adaptive elite fraction
                            mutation_prob=mutation_prob,
                            avoid_inchikeys=seen_inchikeys,
                            component_weights=component_weights,
                        )
                    else:
                        traditional_df = pd.DataFrame(columns=["name", "smiles", "InChIKey"])

                    # Validate and combine
                    if not synthon_df.empty:
                        synthon_df = validate_molecules(synthon_df, config)
                        bt.logging.info(f"[Miner] {len(synthon_df)} synthon candidates passed validation")

                    data = pd.concat([synthon_df, traditional_df], ignore_index=True)
                    data = data.drop_duplicates(subset=["name"], keep="first")
                    bt.logging.info(f"[Miner] Combined: {len(data)} total ({len(synthon_df)} synthon + {len(traditional_df)} GA)")

            elif no_improvement_counter < 3:
                bt.logging.info(f"[Miner] Iteration {iteration}: Standard genetic algorithm")
                data = generate_valid_random_molecules_batch(
                    rxn_id,
                    n_samples=n_samples,
                    db_path=DB_PATH,
                    subnet_config=config,
                    batch_size=dynamic_batch,  # PHASE 2: Dynamic batch size
                    elite_names=elite_names,
                    elite_frac=adaptive_elite_frac,  # PHASE 2: Adaptive elite fraction
                    mutation_prob=mutation_prob,
                    avoid_inchikeys=seen_inchikeys,
                    component_weights=component_weights,
                )

            elif no_improvement_counter < 6:
                bt.logging.info(f"[Miner] Iteration {iteration}: Exploring similar space (no_improvement={no_improvement_counter})")
                data = _cpu_random_candidates_with_similarity(
                    iteration,
                    30,
                    config,
                    top_pool.head(50)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys,
                    0.65
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])

            else:
                bt.logging.info(f"[Miner] Iteration {iteration}: Broad exploration reset (no_improvement={no_improvement_counter})")
                data = _cpu_random_candidates_with_similarity(
                    iteration,
                    40,
                    config,
                    top_pool.head(100)[["name", "smiles", "InChIKey"]],
                    seen_inchikeys,
                    0.0
                )
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])
                no_improvement_counter = 0

            gen_time = time.time() - iter_start_time
            print(f"[Miner] Iteration {iteration}: {len(data)} molecules generated in {gen_time:.1f}s", flush=True)
            bt.logging.info(f"[Miner] Iteration {iteration}: {len(data)} Samples Generated in ~{gen_time:.2f}s (pre-score)")

            if data.empty:
                print(f"[Miner] Iteration {iteration}: NO MOLECULES GENERATED - check rxn:{rxn_id} data", flush=True)
                bt.logging.warning(f"[Miner] Iteration {iteration}: No valid molecules produced; continuing")
                continue

            if not seed_df.empty:
                data = pd.concat([data, seed_df])
                data = data.drop_duplicates(subset=["InChIKey"], keep="first")
                seed_df = pd.DataFrame(columns=["name", "smiles", "InChIKey", "tanimoto_similarity"])

            try:
                filterd_data = data[~data["InChIKey"].isin(seen_inchikeys)]
                if len(filterd_data) < len(data):
                    bt.logging.warning(
                        f"[Miner] Iteration {iteration}: {len(data) - len(filterd_data)} molecules were previously seen"
                    )

                dup_ratio = (len(data) - len(filterd_data)) / max(1, len(data))

                if dup_ratio > 0.7:
                    mutation_prob = min(0.9, mutation_prob * 1.5)
                    elite_frac = max(0.15, elite_frac * 0.7)
                    bt.logging.warning(f"[Miner] SEVERE duplication ({dup_ratio:.2%})! mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                elif dup_ratio > 0.5:
                    mutation_prob = min(0.7, mutation_prob * 1.3)
                    elite_frac = max(0.2, elite_frac * 0.8)
                    bt.logging.warning(f"[Miner] High duplication ({dup_ratio:.2%}), mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                elif dup_ratio < 0.15 and not top_pool.empty and iteration > 10:
                    mutation_prob = max(0.05, mutation_prob * 0.95)
                    elite_frac = min(0.85, elite_frac * 1.05)

                data = filterd_data

            except Exception as e:
                bt.logging.warning(f"[Miner] Pre-score deduplication failed: {e}")

            if data.empty:
                bt.logging.error(f"[Miner] Iteration {iteration}: ALL molecules were duplicates! Skipping scoring and continuing...")
                # Force more diversity for next iteration
                mutation_prob = min(0.95, mutation_prob * 2.0)
                elite_frac = max(0.1, elite_frac * 0.5)
                bt.logging.warning(f"[Miner] Emergency diversity boost: mut={mutation_prob:.2f}, elite={elite_frac:.2f}")
                continue  # Skip to next iteration

            data = data.reset_index(drop=True)

            # Enhanced CPU similarity search - multiple parallel searches
            cpu_futures = []
            if not top_pool.empty and iteration > 3:
                # Multiple parallel CPU searches with different strategies
                if score_improvement_rate < 0.01:
                    # Strategy 1: Tight on top 5
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity,
                            iteration,
                            40,
                            config,
                            top_pool.head(5)[["name", "smiles", "InChIKey"]],
                            seen_inchikeys,
                            0.80
                        ),
                        "tight-top5"
                    ))

                    # Strategy 2: Medium on top 20
                    cpu_futures.append((
                        cpu_executor.submit(
                            _cpu_random_candidates_with_similarity,
                            iteration,
                            30,
                            config,
                            top_pool.head(20)[["name", "smiles", "InChIKey"]],
                            seen_inchikeys,
                            0.65
                        ),
                        "medium-top20"
                    ))

            gpu_start_time = time.time()

            if len(data) == 0:
                bt.logging.error(f"[Miner] Iteration {iteration}: No molecules to score! Continuing...")
                continue

            print(f"[Miner] Iteration {iteration}: Scoring {len(data)} molecules with PSICHIC...", flush=True)
            data["Target"] = target_score_from_data(data["smiles"])
            data["Anti"] = antitarget_scores()
            data["score"] = data["Target"] - (config["antitarget_weight"] * data["Anti"])

            if data["score"].isna().all():
                bt.logging.error(f"[Miner] Iteration {iteration}: Scoring failed (all NaN)! Continuing...")
                continue

            gpu_time = time.time() - gpu_start_time
            print(f"[Miner] Iteration {iteration}: PSICHIC scoring done in {gpu_time:.1f}s", flush=True)
            bt.logging.info(f"[Miner] Iteration {iteration}: GPU scoring time ~{gpu_time:.2f}s")

            # Collect all CPU results
            if cpu_futures:
                for cpu_future, strategy_name in cpu_futures:
                    try:
                        cpu_df = cpu_future.result(timeout=0)
                        if not cpu_df.empty:
                            if seed_df.empty:
                                seed_df = cpu_df.copy()
                            else:
                                seed_df = pd.concat([seed_df, cpu_df], ignore_index=True)
                            bt.logging.info(f"[Miner] CPU similarity ({strategy_name}) found {len(cpu_df)} candidates")
                    except TimeoutError:
                        pass
                    except Exception as e:
                        bt.logging.warning(f"[Miner] CPU similarity ({strategy_name}) failed: {e}")

                if not seed_df.empty:
                    seed_df = seed_df.drop_duplicates(subset=["InChIKey"], keep="first")

            seen_inchikeys.update([k for k in data["InChIKey"].tolist() if k])
            total_data = data[["name", "smiles", "InChIKey", "score", "Target", "Anti"]]
            prev_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            # Safe concatenation
            if not total_data.empty:
                top_pool = pd.concat([top_pool, total_data], ignore_index=True)
                top_pool = top_pool.drop_duplicates(subset=["InChIKey"], keep="first")
                top_pool = top_pool.sort_values(by="score", ascending=False)
            else:
                bt.logging.warning(f"[Miner] Iteration {iteration}: No valid scored data to add to pool")

            # Track best molecules for later intensive exploration
            if not top_pool.empty and iteration % 5 == 0:
                best_molecules_history.append({
                    'iteration': iteration,
                    'molecules': top_pool.head(10)[["name", "smiles", "InChIKey", "score"]].copy()
                })
                if len(best_molecules_history) > 6:
                    best_molecules_history.pop(0)

            # Write top 10 for exploit worker every iteration
            if not top_pool.empty:
                write_top10_for_exploit_worker(top_pool)

            remaining_time = 1800 - (time.time() - start)
            if remaining_time <= 60:
                # FINAL MERGE: Combine Pool A with Pool B from exploit worker
                bt.logging.info("[Miner] === FINAL MERGE PHASE ===")
                pool_b = read_pool_b()
                top_pool = merge_pools(top_pool, pool_b, config)
                entropy = compute_maccs_entropy(top_pool.iloc[:config["num_molecules"]]['smiles'].to_list())
                if entropy > config['entropy_min_threshold']:
                    top_pool = top_pool.head(config["num_molecules"])
                    bt.logging.info(f"[Miner] Iteration {iteration}: Sufficient Entropy = {entropy:.4f}")
                else:
                    try:
                        top_95 = top_pool.iloc[:95]
                        remaining_pool = top_pool.iloc[95:]
                        additional_5 = select_diverse_subset(remaining_pool, top_95["smiles"].tolist(),
                                                            subset_size=5, entropy_threshold=config['entropy_min_threshold'])
                        if not additional_5.empty:
                            top_pool = pd.concat([top_95, additional_5]).reset_index(drop=True)
                            entropy = compute_maccs_entropy(top_pool['smiles'].to_list())
                            bt.logging.info(f"[Miner] Iteration {iteration}: Adjusted Entropy = {entropy:.4f}")
                        else:
                            top_pool = top_pool.head(config["num_molecules"])
                    except Exception as e:
                        bt.logging.warning(f"[Miner] Entropy handling failed: {e}")
            else:
                top_pool = top_pool.head(config["num_molecules"])

            current_avg_score = top_pool['score'].mean() if not top_pool.empty else None

            if current_avg_score is not None:
                if prev_avg_score is not None:
                    score_improvement_rate = (current_avg_score - prev_avg_score) / max(abs(prev_avg_score), 1e-6)
                prev_avg_score = current_avg_score

            if score_improvement_rate == 0.0:
                no_improvement_counter += 1
            else:
                no_improvement_counter = 0

            iter_total_time = time.time() - iter_start_time
            total_time = time.time() - start

            # Pool A stats
            pool_a_avg = top_pool['score'].mean()
            pool_a_max = top_pool['score'].max()
            pool_a_min = top_pool['score'].min() if len(top_pool) >= 100 else top_pool['score'].min()
            try:
                pool_a_entropy = compute_maccs_entropy(top_pool.head(100)['smiles'].tolist())
            except:
                pool_a_entropy = 0.0

            # Pool B stats (read current state) and build combined
            pool_b_avg, pool_b_max, pool_b_entropy, pool_b_size = 0.0, 0.0, 0.0, 0
            combined_avg, combined_max = pool_a_avg, pool_a_max
            combined_pool = top_pool  # Default to Pool A only
            try:
                pool_b_df = read_pool_b()
                if not pool_b_df.empty:
                    pool_b_size = len(pool_b_df)
                    pool_b_avg = pool_b_df['score'].mean()
                    pool_b_max = pool_b_df['score'].max()
                    try:
                        pool_b_entropy = compute_maccs_entropy(pool_b_df['smiles'].tolist())
                    except:
                        pool_b_entropy = 0.0

                    # Combined top 100 (Pool A + Pool B)
                    combined_pool = merge_pools(top_pool.copy(), pool_b_df.copy(), config)
                    combined_avg = combined_pool['score'].mean()
                    combined_max = combined_pool['score'].max()
            except:
                pass

            # Write combined best molecules to result.json
            top_entries = {"molecules": combined_pool["name"].tolist()}

            bt.logging.warning(
                f"Iter {iteration} | {iter_total_time:.1f}s | Total: {total_time:.0f}s | "
                f"PoolA: avg={pool_a_avg:.4f} max={pool_a_max:.4f} ent={pool_a_entropy:.3f} | "
                f"PoolB({pool_b_size}): avg={pool_b_avg:.4f} max={pool_b_max:.4f} ent={pool_b_entropy:.3f} | "
                f"Combined: avg={combined_avg:.4f} max={combined_max:.4f}"
            )

            with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as f:
                json.dump(top_entries, f, ensure_ascii=False, indent=2)

    # PHASE 3: Save warm start for future runs
    save_warm_start_molecules(rxn_id, top_pool)
    
    # Cleanup: stop exploit worker
    stop_exploit_worker()


if __name__ == "__main__":
    args = parse_args()
    config = get_config(input_file=args.input, rxn_override=args.rxn)
    start_time_1 = time.time()
    initialize_models(config)
    bt.logging.info(f"{time.time() - start_time_1} seconds for model initialization")
    main(config)