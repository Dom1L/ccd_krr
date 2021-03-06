import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm


class Compound:
    def __init__(self, xyz_files):
        self.max_natoms = 0
        self.coordinates = []
        self.nuclear_charge = []
        self.representation = []
        self.element_to_charge = {'H': 1,
                                  'C': 6,
                                  'N': 7,
                                  'O': 8,
                                  'F': 9,
                                  'S': 16}
        self.parse_qm7(xyz_files)

    def parse_qm7(self, xyz_files):
        for xyz_file in tqdm(xyz_files):
            coords, elements = self.read_xyz(xyz_file)
            nuclear_charge = np.vectorize(self.element_to_charge.__getitem__)(elements)
            cm_matrix = self.get_coulomb_matrix(coords, nuclear_charge, sorting='norm-row')

            natoms = len(elements)  # get natoms to check later our coordinate size
            self.max_natoms = natoms if natoms > self.max_natoms else self.max_natoms

            self.coordinates.append(coords)
            self.nuclear_charge.append(nuclear_charge)
            self.representation.append(cm_matrix)

    def read_xyz(self, filename):
        with open(filename, 'r') as infile:
            lines = infile.readlines()
        elements = []
        coords = []
        for line in lines[2:]:
            atom, x, y, z = line.split()
            elements.append(atom)
            coords.append([x, y, z])
        return np.array(coords).astype('float'), elements

    def get_coulomb_matrix(self, coords, nuclear_charge, sorting='norm-row', vectorize=True):
        with np.errstate(divide='ignore'):  # ignore the dividing by 0 warning
            inv_dist = 1/self.calculate_distances(coords)
        # First, calculate the off diagonals
        zizj = nuclear_charge[None, :]*nuclear_charge[:, None]
        np.fill_diagonal(inv_dist, 0)  # to get rid of nasty NaNs
        coulomb_matrix = zizj*inv_dist
        # Second, calculate self interaction
        np.fill_diagonal(coulomb_matrix, 0.5 * nuclear_charge ** 2.4)
        if sorting == 'norm-row':
            idx_list = np.argsort(np.linalg.norm(coulomb_matrix, axis=1))
            coulomb_matrix = coulomb_matrix[idx_list][:, idx_list]
        if vectorize:
            triu_idx = np.triu_indices_from(coulomb_matrix)
            coulomb_matrix = coulomb_matrix[triu_idx]

        return coulomb_matrix

    def calculate_distances(self, coordinates):
        return squareform(pdist(coordinates, lambda a, b: np.sqrt(np.sum((a - b) ** 2))))

    def zero_padding(self):
        n_molecules = len(self.representation)
        vec_size, _ = np.triu_indices(self.max_natoms)
        padded_representation = np.zeros((n_molecules, vec_size.shape[0]))
        for i, matrix in enumerate(self.representation):
            n_atoms = matrix.shape[0]
            padded_representation[i, :n_atoms] = matrix
        return padded_representation

    def get_representation(self):
        return self.zero_padding()

    def get_labels(self, energy_file):
        with open(energy_file) as infile:
            lines = infile.readlines()
        return np.array([float(line.split()[1]) for line in lines])


if __name__ == '__main__':
    from glob import glob
    qm7_files = sorted(glob('qm7/xyz/*.xyz'))
    c = Compound(qm7_files)
    representation = c.get_representation()
    energies = c.get_labels('qm7/energies.txt')
