
import operator
ops = {"+": operator.add,
       "-": operator.sub,
       "*": operator.mul,
       "/": operator.truediv}


class Matrix:

    def __init__(self, arg) -> None:
        if (arg is None):
            print("Nothing got in")
            return
        self.data = None
        self.shape = None
        if isinstance(arg, list):
            if all(len(sublist) == len(arg[0]) for sublist in arg):
                self.data = arg
                self.shape = (len(self.data), len(self.data[0]))
            else:
                print("Not all sublists have the same length")
            return
        elif isinstance(arg, tuple):
            self.data = [[0.0 for j in range(arg[0])] for i in range(arg[1])]
            self.shape = arg
            return

    def __getitem__(self, indices):
        if isinstance(indices, tuple):
            row_index, col_index = indices
            return self.data[row_index][col_index]
        elif isinstance(indices, int):
            return self.data[indices]
        else:
            raise TypeError("Invalid index type. Must be int or tuple.")

    def __setitem__(self, indices, value):
        if isinstance(indices, tuple):
            row_index, col_index = indices
            self.data[row_index][col_index] = value
        elif isinstance(indices, int):
            self.data[indices] = value
        else:
            raise TypeError("Invalid index type. Must be int or tuple.")

    def __str__(self):
        return f"Matrix({self.data})"

    def __repr__(self):
        return f"Matrix({self.data})"

    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only matrix of same shape is allowed")

        tmp = []
        # print(self.data[0] + other.data[0], self.shape[0])
        for i in range(0, self.shape[0]):
            tmp.append([a + b for a, b in zip(self.data[i], other.data[i])])
        return Matrix(tmp)

    def __radd__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only matrix of same shape is allowed")
        return other + self

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can subtract from each other")
        if self.shape != other.shape:
            raise ValueError("only matrix of same shape is allowed")

        tmp = []
        for i in range(0, self.shape[0]):
            # print(self.data[i], other.data[i])
            tmp.append([a - b for a, b in zip(self.data[i], other.data[i])])

        return Matrix(tmp)

    def __rsub__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("only matrix can subtract from each other")
        if self.shape != other.shape:
            raise ValueError("only matrix of same shape is allowed")
        return other - self

    def __truediv__(self, var):
        if isinstance(var, Vector):
            raise NotImplementedError(
                "Division of a Matrix by a Matrix is not implemented here.")
        if not any([isinstance(var, t) for t in [float, int, complex]]):
            raise ValueError("division only accepts scalar. (real number)")
        if var == 0:
            raise ValueError("Division of 0 not allowed.")
        tmp = []
        for i in range(0, self.shape[0]):
            # print(self.data[i], other.data[i])
            tmp.append([a / var for a in self.data[i]])
        return Matrix(tmp)

    def __rtruediv__(self, var):
        raise NotImplementedError("rtruediv not implemented")

    def __mul__(self, var):
        # print(type(var))
        if any(isinstance(var, scalar_type) for scalar_type in [int, float, complex]):
            result = [[self.data[i][j] *
                       var for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Matrix(result)
        elif isinstance(var, Vector):
            if self.shape[1] != var.shape[0]:
                raise ValueError(
                    "Matrices cannot be multiplied, dimensions don't match.")
            result = [[sum([self.data[i][k] * var.data[k][j] for k in range(self.shape[1])])
                       for j in range(var.shape[1])] for i in range(self.shape[0])]
            return Vector(result)
        elif isinstance(var, Matrix):
            # print("b", self.shape[1], var.shape[0])
            if self.shape[1] != var.shape[0]:
                raise ValueError(
                    "Matrices cannot be multiplied, dimensions don't match.")
            result = [[sum([self.data[i][k] * var.data[k][j] for k in range(self.shape[1])])
                       for j in range(var.shape[1])] for i in range(self.shape[0])]
            return Matrix(result)
        else:
            raise TypeError("Invalid type of input value.")

    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError(
                "unsupported operand type(s) for +: '{}' and '{}'".format(type(self), type(other)))
        if self.shape[1] != other.shape[0]:
            raise TypeError(
                "Invalid input: dot product requires a Vector of compatible shape.")
        result = 0.0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result += self.data[i][j] * other.data[j][i]
        return result

    def __rmul__(self, x):
        return self * x

    # ref: https://stackoverflow.com/questions/21444338/transpose-nested-list-in-python

    def T(self):
        return Matrix(list(map(list, zip(*self.data))))

    # For use in `matrix` project
    # ex00 - add, sub, scl
    def add(self, other):
        return self.__add__(other)

    def sub(self, other):
        return self.__sub__(other)

    def scl(self, number):
        if not any(isinstance(number, scalar_type) for scalar_type in [int, float, complex]):
            raise TypeError("scl() function only accepts numerical type")
        # as it is already implemented in __mul__
        return self.__mul__(number)


class Vector(Matrix):

    def __init__(self, data):
        # a list of a list of floats: Vector([[0.0, 1.0, 2.0, 3.0]]),
        # • a list of lists of single float: Vector([[0.0], [1.0], [2.0], [3.0]]),
        # • a size: Vector(3) -> the vector will have values = [[0.0], [1.0], [2.0]],
        # • a range: Vector((10,16)) -> the vector will have values = [[10.0], [11.0],
        #            [12.0], [13.0], [14.0], [15.0]]. in Vector((a,b)), if a > b, you must display accurate error message
        ##
        if (isinstance(data, int)):
            if (data < 0):
                raise ValueError(
                    "Vector must be initialized with appropriate data (int, negative)")
            self.data = []
            for i in range(data):
                self.data.append([float(i)])
        elif (isinstance(data, tuple)):
            if not (len(data) == 2):
                raise ValueError(
                    "Vector must be initialized with appropriate data (tuple, length)")
            if not (isinstance(data[0], int) and isinstance(data[1], int)):
                raise ValueError(
                    "Vector must be initialized with appropriate data (tuple, data type)")
            if not (data[0] < data[1]):
                raise ValueError(
                    "Vector must be initialized with appropriate data (tuple, range)")
            self.data = []
            for i in range(data[0], data[1]):
                self.data.append([float(i)])
        # elif not (any(isinstance(i, list) for i in data) and isinstance(data, list)):
            # raise TypeError("vector must be initialized with appropriate data")
        else:
            # for list_inside in data:
            #     if isinstance(list_inside, list):
            #         for j in list_inside:
            #             if not (isinstance(j, float)):
            #                 raise TypeError(
            #                     "The element must be float type")
            #     else:
            #         if not (isinstance(list_inside, float)):
            #             raise TypeError("The element must be float type")

            self.data = data

        if len(self.data) == 1:
            # print(len(data[0]))
            self.shape = (1, len(self.data[0]))
        else:
            # print(len(data))
            self.shape = (len(self.data), 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __str__(self):
        return f"Vector({self.data})"

    def __repr__(self):
        return f"Vector({self.data})"

    def _T_row_to_col(self):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append([self.data[0][i]])
        return Vector(tmp)

    def _T_col_to_row(self):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append(self.data[i][0])
        return Vector([tmp])

    def T(self):
        dimension_check = self.shape.index(max(self.shape))
        if (self.shape == (1, 1)):
            return self
        if dimension_check == 1:
            return self._T_row_to_col()
        else:
            return self._T_col_to_row()

    # ex03 - alerady implemented
    def dot(self, other):
        if not isinstance(other, Vector):
            raise TypeError("unsupported operand type(s) for dot product: '{}' and '{}'".format(
                type(self), type(other)))

        if self.shape[0] != other.shape[0]:
            raise ValueError(
                "Invalid input: dot product requires vectors of the same size.")

        result = 0.0
        for i in range(self.shape[0]):
            result += self.data[i][0] * other.data[i][0]

        return result

    def __add__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("only vector can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only vector of same shape is allowed")
        dimension_check = self.shape.index(max(self.shape))
        tmp = []
        if dimension_check == 1:
            for i in range(0, max(self.shape)):
                tmp.append(self.data[0][i] + other.data[0][i])
            return Vector([tmp])
        else:
            for i in range(0, max(self.shape)):
                tmp.append([self.data[i][0] + other.data[i][0]])
            return Vector(tmp)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        if not isinstance(other, Vector):
            raise TypeError("only vector can add to each other")
        if (self.shape != other.shape):
            raise ValueError("only vector of same shape is allowed")
        dimension_check = self.shape.index(max(self.shape))
        tmp = []
        if dimension_check == 1:
            for i in range(0, max(self.shape)):
                tmp.append(self.data[0][i] - other.data[0][i])
            return Vector([tmp])
        else:
            for i in range(0, max(self.shape)):
                tmp.append([self.data[i][0] - other.data[i][0]])
            return Vector(tmp)

    def __rsub__(self, other):
        return other - self

    def __row_loop(self, var, operator):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append(ops[operator](self.data[0][i], var))
        return Vector([tmp])

    def __col_loop(self, var, operator):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append([ops[operator](self.data[i][0], var)])
        return Vector(tmp)

    def col_loop(self, other, operator):
        tmp = []
        for i in range(0, max(self.shape)):
            tmp.append([ops[operator](self.data[0][i], other.data[0][i])])
        return Vector([tmp])

    def __truediv__(self, var):
        if isinstance(var, Vector):
            raise NotImplementedError(
                "Division of a Vector by a Vector is not implemented here.")
        if not any([isinstance(var, t) for t in [float, int, complex]]):
            raise ValueError("division only accepts scalar. (real number)")
        dimension_check = self.shape.index(max(self.shape))
        if dimension_check == 1:
            return self.__row_loop(var, "/")
        else:
            return self.__col_loop(var, "/")

    def __rtruediv__(self, var):
        raise NotImplementedError(
            "Division of a scalar by a Vector is not implemented here.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # Multiply each element of the vector by a scalar
            result = Vector([[element * other] for element in self])
            return result
        elif isinstance(other, Vector):
            # Multiply two vectors element-wise
            if len(self) != len(other):
                raise ValueError(
                    "Vectors must have the same length for element-wise multiplication")
            result = Vector([[a * b] for a, b in zip(self, other)])
            return result
        elif isinstance(other, Matrix):
            # Multiply a vector by a matrix
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    "Matrix column count must match vector row count for multiplication")
            result = Matrix([[sum(a * b for a, b in zip(row, self))]
                            for row in other])
            return result
        else:
            raise TypeError("Unsupported operand types for multiplication")

    # ex04 - norm_1(), norm(), norm_inf()
    def norm_1(self):
        ret = 0
        for i in range(0, max(self.shape)):
            ret += (self.data[i][0] if self.data[i]
                    [0] > 0 else self.data[i][0] * -1)
        return ret

    def norm(self):
        ret = 0
        for i in range(0, max(self.shape)):
            ret += (self.data[i][0] ** 2)
        return ret ** (1/2)

    def norm_inf(self):
        flatten_list = [item for sublist in self.data for item in sublist]
        absolute_list = [(x if x > 0 else x * -1) for x in flatten_list]
        return max(absolute_list)


def create_zero_vectors(num_vectors):
    zero_vector = [0]
    zero_vectors = [zero_vector] * num_vectors
    return zero_vectors


def create_zero_matrix(rows, columns):
    zero_matrix = [[0.0] * columns for _ in range(rows)]
    return zero_matrix

# ex01 - linear combination


def linear_combination(sets, coeffs):
    # Error handling for sets and coeffs parameters
    if not isinstance(sets, list) or not isinstance(coeffs, list):
        raise TypeError("Both sets and coeffs must be passed as lists.")

    if not all(isinstance(item, Vector) for item in sets):
        raise ValueError(
            "All elements in sets must be instances of Vector class.")

    if not all(isinstance(coeff, (int, float, complex)) for coeff in coeffs):
        raise ValueError("Coefficients can only be numerical types.")

    if len(sets) != len(coeffs):
        raise ValueError("Number of sets and coefficients must match.")

    ret = Vector(create_zero_vectors(len(sets)))
    for i in range(len(sets)):
        ret += coeffs[i] * sets[i]

    return ret

# ex02 - linear interpolation
# be careful with decimal_place !!


def lerp(u, v, t, decimal_place=1):
    if isinstance(u, Vector):
        if not isinstance(v, Vector):
            raise TypeError(
                "Instances u and v must both be instances of Vector.")
    elif isinstance(u, Matrix):
        if not isinstance(v, Matrix):
            raise TypeError(
                "Instances u and v must both be instances of Matrix.")
    elif isinstance(u, (int, float, complex)):
        if not isinstance(v, (int, float, complex)):
            raise TypeError(
                "Instances u and v must have the same type when u is a scalar.")
    else:
        raise TypeError(
            "Instances u and v must be instances of Vector or Matrix class, or numerical types.")

    if t == 0:
        return u
    elif t == 1:
        return v

    if isinstance(u, (int, float, complex)):
        return round((1 - t) * u + t * v, decimal_place)
    # Perform linear interpolation between u and v
    elif isinstance(u, Vector):
        if u.shape != v.shape:
            raise ValueError(
                "Vectors must have the same shape, but received vectors of different shapes.")
        interpolated = Vector(create_zero_vectors(u.shape[0]))
        for i in range(u.shape[0]):
            interpolated[i] = round(
                (1 - t) * u[i][0] + t * v[i][0], decimal_place)
    elif isinstance(u, Matrix):
        if u.shape != v.shape:
            raise ValueError(
                "Matrices must have the same shape, but received matrices of different shapes.")
        num_rows, num_cols = u.shape[0], u.shape[1]
        interpolated = Matrix(create_zero_matrix(num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                interpolated[i][j] = (1 - t) * u[i][j] + t * v[i][j]
    else:
        raise TypeError(
            "Instances u and v must be instances of Vector or Matrix class.")

    return interpolated


# ex05 - angle_cos
# be careful with decimal_place !!

def angle_cos(u: Vector, v: Vector,  decimal_place=1):
    dot = u.dot(v)
    mag1 = u.norm()
    mag2 = v.norm()

    if mag1 == 0 or mag2 == 0:
        return float('nan')  # Return NaN for zero vectors

    cos_theta = dot / (mag1 * mag2)
    return round(cos_theta, decimal_place) if decimal_place == 1 else cos_theta
