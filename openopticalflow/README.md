# OpenOpticalFlow

This directory contains Python implementations of optical flow algorithms and related utilities for fluid flow analysis.

## New Additions

The following files have been added from the comparison implementation:

1. **preprocessing.py** - A comprehensive preprocessing module that includes:
   - `pre_processing()` - Resize and filter images
   - `correction_illumination()` - Correct illumination differences between images
   - `shift_image_refine()` - Shift image based on velocity field

## Tests

All test files have been moved to the `tests/` directory:

1. **tests/test_vorticity_example.py** - A simple example test showing how to use the vorticity_factor function.

2. **tests/test_validation.py** - A comprehensive test for validating the optical flow implementation with synthetic and real image pairs.

3. **tests/test_horn_schunck.py** - A test specifically for the Horn-Schunck estimator.

## Usage

To run the example test:

```python
python tests/test_vorticity_example.py
```

To run the validation test:

```python
python tests/test_validation.py
```

To test the Horn-Schunck estimator:

```python
python tests/test_horn_schunck.py
```

## Dependencies

- NumPy
- SciPy
- Matplotlib
- scikit-image (for image I/O in validation tests)

## Notes

- The `preprocessing.py` module provides a more comprehensive set of preprocessing functions compared to the original `pre_processing_a.py`.
- For backward compatibility, `pre_processing_a` is still available as an alias for `pre_processing`.
- The validation test creates synthetic test cases if real image pairs are not available.
