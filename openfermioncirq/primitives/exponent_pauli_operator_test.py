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

import unittest

import cirq
import numpy
import scipy
from openfermion import QubitOperator, get_sparse_operator

from openfermioncirq.primitives import (pauli_exponent_to_circuit,
                                        trotter_qubitoperator_exponent)


class ExponentPauliOperatorTest(unittest.TestCase):
    """Test class of Exponent Pauli Operator."""

    def test_function_raise(self):
        """Test function raises."""
        op1 = QubitOperator('X0', 1.0)
        op2 = QubitOperator('X0 Y1', numpy.pi / 2)
        qubit_list = [cirq.LineQubit(q) for q in range(4)]
        param_list = [0.0]

        with self.assertRaises(TypeError):
            cirq.Circuit(pauli_exponent_to_circuit(1.0, qubit_list, None))
        with self.assertRaises(ValueError):
            cirq.Circuit(pauli_exponent_to_circuit(
                op1 + op2, qubit_list, None))
        with self.assertRaises(ValueError):
            cirq.Circuit(pauli_exponent_to_circuit(op1, qubit_list, None))
        with self.assertRaises(TypeError):
            cirq.Circuit(trotter_qubitoperator_exponent(1.0, qubit_list, None))
        with self.assertRaises(ValueError):
            cirq.Circuit(trotter_qubitoperator_exponent(op1 + op2, qubit_list,
                                                        param_list))

    def test_unitary_circuit(self):
        """Test circuit matrix and operator are equal."""
        op1 = QubitOperator('X0 X1', numpy.random.rand() * numpy.pi)
        op1mat = scipy.linalg.expm(-0.5j * get_sparse_operator(op1, 2))
        qubit_list = [cirq.LineQubit(q) for q in range(4)]

        circuit = cirq.Circuit(pauli_exponent_to_circuit(op1,
                                                         qubit_list, None))
        self.assertTrue(numpy.allclose(op1mat.toarray(), circuit.unitary()))

    def test_unitary_circuit_multiple_paulis(self):
        """Test matrix with multiple Pauli operators."""
        op2 = (QubitOperator('X0 X1', 0.5) +
               QubitOperator('Z0 Z1 Y2 Y3', numpy.pi) +
               QubitOperator('X0 Y1 Y2 X3', numpy.pi / 2) +
               QubitOperator('Z0 X1 X2 Y3', -1 * numpy.pi / 2))
        op2mat = numpy.identity(2**4)

        qubit_list = [cirq.LineQubit(q) for q in range(4)]

        circuit = cirq.Circuit(
            trotter_qubitoperator_exponent(op2, qubit_list, None))
        for op in reversed(list(op2)):
            op2mat *= scipy.linalg.expm(-0.5j * get_sparse_operator(op, 4))

        self.assertTrue(numpy.allclose(op2mat, circuit.unitary()))
