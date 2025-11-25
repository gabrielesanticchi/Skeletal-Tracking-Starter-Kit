"""
Convert SMPL model from chumpy format to pure numpy format.

This allows using SMPL models without having to install the deprecated chumpy package.
"""

import pickle
import numpy as np
from pathlib import Path
import sys


class ChumpyUnpickler(pickle.Unpickler):
    """Custom unpickler that converts chumpy arrays to numpy arrays."""

    def find_class(self, module, name):
        # Redirect chumpy classes to numpy
        if 'chumpy' in module:
            # Return a class that behaves like numpy array
            return lambda *args, **kwargs: np.array(args[0]) if args else np.array([])
        return super().find_class(module, name)


def convert_smpl_model(input_path: Path, output_path: Path):
    """
    Convert SMPL model from chumpy to numpy format.

    Args:
        input_path: Path to original SMPL model (with chumpy)
        output_path: Path to save converted model
    """
    print(f'Loading {input_path}...')

    with open(input_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    print(f'Loaded {len(data)} keys')

    # Convert all values to pure numpy/python types
    output_dict = {}
    for key, value in data.items():
        if hasattr(value, '__array__'):
            output_dict[key] = np.array(value)
        else:
            output_dict[key] = value

        if isinstance(output_dict[key], np.ndarray):
            print(f'  {key}: {output_dict[key].shape} {output_dict[key].dtype}')

    print(f'\nSaving to {output_path}...')
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)

    print('✓ Conversion complete!')

    # Test loading
    print('\nVerifying converted model...')
    with open(output_path, 'rb') as f:
        test_data = pickle.load(f)
    print(f'✓ Successfully loaded {len(test_data)} keys')

    return output_dict


if __name__ == '__main__':
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
                 Path(__file__).parent.parent.parent / 'models' / 'smpl' / 'SMPL_NEUTRAL.pkl'

    output_path = input_path.parent / f'{input_path.stem}_converted.pkl'

    convert_smpl_model(input_path, output_path)
