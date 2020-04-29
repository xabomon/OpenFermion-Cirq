#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Test function of exponent pauli operator"""

import cirq
import numpy
import scipy
from openfermion import QubitOperator, get_sparse_operator

import pytest

from openfermioncirq.primitives import (pauli_exponent_to_circuit)


def test_function_raise():
    """Test function raises."""
    op1 = QubitOperator('X0', 1.0)
    op2 = QubitOperator('X0 Y1', numpy.pi / 2)
    qubit_list = [cirq.LineQubit(q) for q in range(4)]
    param_list = [0.0]

    with pytest.raises(TypeError):
        cirq.Circuit(pauli_exponent_to_circuit(1.0, qubit_list, None))
    with pytest.raises(ValueError):
        cirq.Circuit(pauli_exponent_to_circuit(
            op1 + op2, qubit_list, None))
    with pytest.raises(ValueError):
        cirq.Circuit(pauli_exponent_to_circuit(op1, qubit_list, None))


def test_identity_circuit():
    """Test exponent with 0 coefficient generates Identity."""
    op = QubitOperator('X0 Y1', 1.0)
    qubit_list = [cirq.LineQubit(q) for q in range(2)]

    circuit = cirq.Circuit(
        pauli_exponent_to_circuit(op * 0.0, qubit_list, None))

    numpy.testing.assert_allclose(numpy.identity(2**len(qubit_list)),
                                  circuit.unitary(),
                                  rtol=1e-7, atol=1e-7)


def test_unitary_circuit():
    """Test circuit matrix and operator are equal."""
    op1 = QubitOperator('X0 X1', numpy.random.rand() * numpy.pi)
    op1mat = scipy.linalg.expm(-0.5j * get_sparse_operator(op1, 2))
    qubit_list = [cirq.LineQubit(q) for q in range(4)]

    circuit = cirq.Circuit(pauli_exponent_to_circuit(op1,
                                                     qubit_list, None))
    numpy.testing.assert_allclose(op1mat.toarray(), circuit.unitary(),
                                  rtol=1e-7, atol=1e-7)
