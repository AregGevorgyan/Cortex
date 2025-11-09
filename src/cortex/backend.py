"""
Backend abstraction layer for Cortex.

Allows switching between NumPy (CPU) and CuPy (GPU) computation
while keeping all tensor operations unchanged.
"""

# Global backend state
_backend_name = 'numpy'
_backend_module = None


def set_backend(backend='numpy'):
    """
    Set the computation backend.

    Parameters
    ----------
    backend : str
        Backend to use. Options:
        - 'numpy' or 'cpu': Use NumPy for CPU computation (default)
        - 'cupy' or 'gpu': Use CuPy for GPU computation

    Returns
    -------
    module
        The backend module (numpy or cupy)

    Raises
    ------
    ImportError
        If CuPy is requested but not installed
    ValueError
        If an unknown backend is specified

    Examples
    --------
    >>> import cortex
    >>> cortex.set_backend('cpu')  # Use NumPy
    >>> cortex.set_backend('gpu')  # Use CuPy
    """
    global _backend_name, _backend_module

    # Normalize backend name
    backend = backend.lower()
    if backend in ('numpy', 'cpu'):
        backend = 'numpy'
    elif backend in ('cupy', 'gpu'):
        backend = 'cupy'
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Valid options: 'numpy', 'cpu', 'cupy', 'gpu'"
        )

    # Import the appropriate module
    if backend == 'cupy':
        try:
            import cupy as cp
            _backend_module = cp
            _backend_name = 'cupy'
            print(f"Backend set to CuPy (GPU). Device: {cp.cuda.Device()}")
        except ImportError:
            raise ImportError(
                "CuPy is not installed. To use GPU backend, install CuPy:\n"
                "  For CUDA 11.x: pip install cupy-cuda11x\n"
                "  For CUDA 12.x: pip install cupy-cuda12x\n"
                "  For ROCm: pip install cupy-rocm-x-y\n"
                "  See: https://docs.cupy.dev/en/stable/install.html"
            )
    else:  # numpy
        import numpy as np
        _backend_module = np
        _backend_name = 'numpy'

    return _backend_module


def get_backend():
    """
    Get the current backend module.

    Returns
    -------
    module
        The current backend module (numpy or cupy)
    str
        The backend name ('numpy' or 'cupy')

    Examples
    --------
    >>> import cortex
    >>> np, name = cortex.get_backend()
    >>> print(name)
    'numpy'
    """
    global _backend_module, _backend_name

    # Lazy initialization to numpy if not set
    if _backend_module is None:
        set_backend('numpy')

    return _backend_module, _backend_name


def get_backend_name():
    """
    Get the name of the current backend.

    Returns
    -------
    str
        'numpy' for CPU backend, 'cupy' for GPU backend
    """
    _, name = get_backend()
    return name


# Module-level np that can be imported by other modules
# This gets dynamically set to either numpy or cupy
np, _ = get_backend()


def to_numpy(array):
    """
    Convert array to NumPy array (CPU).

    Useful for moving data from GPU to CPU for visualization, saving, etc.

    Parameters
    ----------
    array : ndarray
        NumPy or CuPy array

    Returns
    -------
    ndarray
        NumPy array on CPU

    Examples
    --------
    >>> import cortex
    >>> cortex.set_backend('gpu')
    >>> x = cortex.Tensor([1, 2, 3])
    >>> x_cpu = cortex.backend.to_numpy(x.data)
    """
    backend_module, backend_name = get_backend()

    if backend_name == 'cupy':
        # CuPy array -> NumPy array
        if hasattr(array, 'get'):
            return array.get()
        elif hasattr(array, 'asnumpy'):
            return array.asnumpy()
        else:
            # Already numpy or scalar
            return array
    else:
        # Already NumPy
        return array


def to_backend(array):
    """
    Convert array to current backend (NumPy or CuPy).

    Parameters
    ----------
    array : ndarray or array-like
        Array to convert

    Returns
    -------
    ndarray
        Array in current backend (NumPy or CuPy)

    Examples
    --------
    >>> import cortex
    >>> import numpy as np
    >>> cortex.set_backend('gpu')
    >>> x_cpu = np.array([1, 2, 3])
    >>> x_gpu = cortex.backend.to_backend(x_cpu)
    """
    backend_module, backend_name = get_backend()

    if backend_name == 'cupy':
        # Ensure it's a CuPy array
        if hasattr(array, '__cuda_array_interface__'):
            # Already on GPU
            return array
        else:
            # Move to GPU
            return backend_module.asarray(array)
    else:
        # Ensure it's a NumPy array
        import numpy as numpy_module
        return numpy_module.asarray(array)
