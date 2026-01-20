"""Cache Radia magnetization solutions for faster re-solving."""

import pickle
import os
import hashlib
from typing import Dict, Optional, Tuple
import numpy as np


class RadiaCache:
    """Cache Radia magnetization states between runs."""

    def __init__(self, cache_dir: str = 'output/radia_cache', rank: int = 0, verbosity: int = 0):
        """
        Initialize cache manager.

        :param cache_dir: Directory to store cache files
        :param rank: MPI rank
        :param verbosity: Verbosity level
        """
        self.cache_dir = cache_dir
        self.rank = rank
        self.verbosity = verbosity
        os.makedirs(cache_dir, exist_ok=True)

    def compute_parameter_signature(self,
                                    config) -> str:
        """
        Compute hash of all parameters that invalidate the cache.

        If any of these change, cached magnetizations are unusable:
        - Coil current
        - Yoke wall parameters
        - Lid parameters
        - Pole base parameters

        :param config: CyclotronConfig
        :return: Hash string
        """

        params_dict = {
            'coil_current_A': config.coil.current_A,
            'yoke_wall_height_mm': config.yoke.height_mm,
            'yoke_wall_inner_radius_mm': config.yoke.inner_radius_mm,
            'yoke_wall_outer_radius_mm': config.yoke.outer_radius_mm,
            'lid_lower_height_mm': config.lid_lower.height_mm,
            'lid_lower_inner_radius_mm': config.lid_lower.inner_radius_mm,
            'lid_lower_outer_radius_mm': config.lid_lower.outer_radius_mm,
            'pole_base_inner_radius_mm': config.pole.inner_radius_mm,
            'pole_base_outer_radius_mm': config.pole.outer_radius_mm,
            'pole_base_height_mm': config.pole.height_mm,
        }

        # Create deterministic string and hash it
        param_str = str(sorted(params_dict.items()))
        signature = hashlib.md5(param_str.encode()).hexdigest()

        if self.rank <= 0 and self.verbosity >= 2:
            print(f"[RadiaCache] Parameter signature: {signature}", flush=True)

        return signature

    def save_magnetizations(self,
                            config,
                            magnetizations: Dict[int, Tuple[float, float, float]],
                            cache_name: str = 'magnetizations') -> str:
        """
        Save magnetization states to cache.

        :param config: CyclotronConfig (used to compute signature)
        :param magnetizations: Dict mapping Radia ID → (Mx, My, Mz)
        :param cache_name: Base name for cache file
        :return: Path to cache file
        """

        signature = self.compute_parameter_signature(config)
        cache_file = os.path.join(self.cache_dir, f'{cache_name}_{signature}.pkl')

        cache_data = {
            'signature': signature,
            'magnetizations': magnetizations,
            'n_entries': len(magnetizations)
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        if self.rank <= 0 and self.verbosity >= 1:
            print(f"[RadiaCache] Saved {len(magnetizations)} magnetizations to {cache_file}",
                  flush=True)

        return cache_file

    def load_magnetizations(self,
                            config,
                            cache_name: str = 'magnetizations') -> Optional[Dict[int, Tuple[float, float, float]]]:
        """
        Load cached magnetizations if available and valid.

        :param config: CyclotronConfig (used to verify signature)
        :param cache_name: Base name for cache file
        :return: Dict of magnetizations, or None if cache invalid/missing
        """

        signature = self.compute_parameter_signature(config)
        cache_file = os.path.join(self.cache_dir, f'{cache_name}_{signature}.pkl')

        if not os.path.exists(cache_file):
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"[RadiaCache] No cache found for signature {signature}", flush=True)
            return None

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            if cache_data['signature'] != signature:
                if self.rank <= 0 and self.verbosity >= 1:
                    print(f"[RadiaCache] Cache signature mismatch, discarding", flush=True)
                return None

            magnetizations = cache_data['magnetizations']
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"[RadiaCache] Loaded {len(magnetizations)} magnetizations from cache",
                      flush=True)

            return magnetizations

        except Exception as e:
            if self.rank <= 0 and self.verbosity >= 1:
                print(f"[RadiaCache] Failed to load cache: {e}", flush=True)
            return None

    def apply_magnetizations(self,
                             magnetizations: Dict[int, Tuple[float, float, float]]):
        """
        Apply cached magnetizations to Radia objects as starting values.

        :param magnetizations: Dict mapping Radia ID → (Mx, My, Mz)
        """
        import radia as rad

        n_applied = 0
        for obj_id, ((_, _, _), (mx, my, mz)) in magnetizations.items():
            try:
                # Set magnetization as starting value
                rad.ObjSetM(obj_id, [mx, my, mz])
                n_applied += 1
            except Exception as e:
                if self.rank <= 0 and self.verbosity >= 2:
                    print(f"[RadiaCache] Failed to apply magnetization to ID {obj_id}: {e}",
                          flush=True)

        if self.rank <= 0 and self.verbosity >= 1:
            print(f"[RadiaCache] Applied {n_applied} magnetizations", flush=True)
