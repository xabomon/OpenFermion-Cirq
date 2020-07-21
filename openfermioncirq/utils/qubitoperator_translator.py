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

import cirq
from openfermion import QubitOperator

rotation_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}


def _qubitoperator_to_pauli_string(qubit_op: QubitOperator) -> cirq.PauliString:
    """
    Convert QubitOperator to Pauli String.

    Args:
        qubit_op (QubitOperator): operator to convert.

    Returns:
        pauli_string (PauliString): cirq PauliString object.

    Raises:
        TypeError: if qubit_op is not a QubitOpertor.
        ValueError: if qubit_op has more than one Pauli string.
    """
    if not isinstance(qubit_op, QubitOperator):
        raise TypeError('Input must be a QubitOperator.')
    if len(qubit_op.terms) > 1:
        raise ValueError('Input has more than one Pauli string.')

    pauli_string = cirq.PauliString()
    ind_ops, coeff = next(iter(qubit_op.terms.items()))

    if ind_ops == ():
        return pauli_string * coeff

    else:
        for ind, op in ind_ops:

            pauli_string *= rotation_dict[op](cirq.LineQubit(ind))

    return pauli_string * coeff


def qubitoperator_to_pauli_sum(qubit_op: QubitOperator) -> cirq.PauliSum:
    """
    Convert QubitOperator to PauliSum object.

    Args:
        qubit_op (QubitOperator): operator to convert.

    Returns:
        pauli_sum (PauliSum): cirq PauliSum object.

    Raises:
        TypeError: if qubit_op is not a QubitOpertor.
    """
    if not isinstance(qubit_op, QubitOperator):
        raise TypeError('Input must be a QubitOperator.')

    pauli_sum = cirq.PauliSum()
    for pauli in qubit_op:
        pauli_sum += _qubitoperator_to_pauli_string(pauli)

    return pauli_sum
