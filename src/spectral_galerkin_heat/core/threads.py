"""CPU thread control for the solver's three parallel libraries.

The same thread count applies to:

* Numba (``prange`` kernels), via :func:`numba.set_num_threads`;
* scipy.fft (the DCTs), whose calls pass ``workers=numba.get_num_threads()``;
* BLAS (behind ``numpy.einsum``/``tensordot``), via threadpoolctl.
"""

# Copyright 2026 Laboratoire de Mécanique des Solides (LMS),
# École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris,
# Route de Saclay, Palaiseau, 91128, France.
#
# Author: Théo Andrieux, Jules Dichamp, Manas V. Upadhyay
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__author__ = "Théo Andrieux, Jules Dichamp, Manas V. Upadhyay"
__copyright__ = "Copyright 2026 Laboratoire de Mécanique des Solides (LMS), École Polytechnique, CNRS UMR 7649, Institut Polytechnique de Paris"

import logging

import numba
from threadpoolctl import threadpool_limits

logger = logging.getLogger(__name__)

# Kept so the BLAS limit object is not garbage collected.
_blas_limit = None


def configure_threads(threads: int | None = 1) -> int:
    """Set the number of CPU threads used by Numba, the DCTs and BLAS.

    Parameters
    ----------
    threads : int or None
        Number of threads, by default 1 so that a run never takes over a
        shared machine by accident. ``None`` uses every thread Numba was
        started with (all cores unless ``NUMBA_NUM_THREADS`` says otherwise).

    Returns
    -------
    int
        The number of threads in use.
    """
    global _blas_limit
    available = numba.config.NUMBA_NUM_THREADS
    if threads is None:
        threads = available
    threads = int(threads)
    if threads < 1:
        raise ValueError(f"simulation.threads must be >= 1 or 'all', got {threads}.")
    if threads > available:
        logger.warning(
            "simulation.threads=%d exceeds the %d threads Numba was started "
            "with; using %d.", threads, available, available)
        threads = available
    numba.set_num_threads(threads)
    _blas_limit = threadpool_limits(limits=threads, user_api="blas")
    return threads
