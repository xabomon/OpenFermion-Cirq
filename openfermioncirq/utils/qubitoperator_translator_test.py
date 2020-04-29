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

"""Test qubitoperator translator."""

import numpy
import openfermion
import cirq
from openfermion import QubitOperator

from openfermioncirq.utils import qubitoperator_to_pauli_sum
from openfermioncirq.utils.qubitoperator_translator import (
    _qubitoperator_to_pauli_string)

import pytest


def test_function_raises():
    """Test function raises."""
    qop = QubitOperator('X0 Y1', 1.0) + QubitOperator('Z0 Z1', -0.5)
    with pytest.raises(TypeError):
        _qubitoperator_to_pauli_string(1.0)
    with pytest.raises(ValueError):
        _qubitoperator_to_pauli_string(qop)
    with pytest.raises(TypeError):
        qubitoperator_to_pauli_sum([5.0])


@pytest.mark.parametrize(
    'qubitop, state_binary',
    [(QubitOperator('Z0 Z1', -1.0), '00'),
     (QubitOperator('X0 Y1', 1.0), '10')
     ])
def test_expectation_values(qubitop, state_binary):
    """Test PauliSum and QubitOperator expectation value."""
    n_qubits = openfermion.count_qubits(qubitop)
    state = numpy.zeros(2**n_qubits, dtype='complex64')
    state[int(state_binary, 2)] = 1.0
    qubit_map = {cirq.LineQubit(i): i for i in range(n_qubits)}

    pauli_str = _qubitoperator_to_pauli_string(qubitop)
    op_mat = openfermion.get_sparse_operator(qubitop, n_qubits)

    expct_qop = openfermion.expectation(op_mat, state)
    expct_pauli = pauli_str.expectation_from_wavefunction(state, qubit_map)

    numpy.testing.assert_allclose(expct_qop, expct_pauli)


@pytest.mark.parametrize(
    'qubitop, state_binary',
    [(QubitOperator('Z0 Z1 Z2 Z3', -1.0) +
      QubitOperator('X0 Y1 Y2 X3', 1.0), '1100'),
     (QubitOperator('X0 X3', -1.0) +
      QubitOperator('Y1 Y2', 1.0), '0000')
     ])
def test_expectation_values_paulisum(qubitop, state_binary):
    """Test PauliSum and QubitOperator expectation value."""
    n_qubits = openfermion.count_qubits(qubitop)
    state = numpy.zeros(2**n_qubits, dtype='complex64')
    state[int(state_binary, 2)] = 1.0
    qubit_map = {cirq.LineQubit(i): i for i in range(n_qubits)}

    pauli_str = qubitoperator_to_pauli_sum(qubitop)
    op_mat = openfermion.get_sparse_operator(qubitop, n_qubits)

    expct_qop = openfermion.expectation(op_mat, state)
    expct_pauli = pauli_str.expectation_from_wavefunction(state, qubit_map)

    numpy.testing.assert_allclose(expct_qop, expct_pauli)
