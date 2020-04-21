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

"""Exponential of Pauli operators"""

from typing import List, Tuple, Sequence, Optional

import cirq
import numpy

from openfermion import QubitOperator
import sympy

# Dictionary of single qubit pre-rotaitons.

rot_dic = {'X': lambda q, s: cirq.ry(s * numpy.pi / 2)(q),
           'Y': lambda q, s: cirq.rx(s * numpy.pi / 2)(q),
           'Z': lambda q, s: cirq.I(q)}


def pauli_exponent_to_circuit(
        pauli: QubitOperator,
        qubits: Sequence[cirq.Qid],
        parameter: None) -> List[cirq.Operation]:
    r"""
    Convert Pauli string to a unitary circuit by exponentiation.

    Pauli operators can be converted to a unitary matrix,
    :math:`U(\theta) = e^{i \theta P}`,
    and then into a circuit by using single-qubit rotations and CNOTs.

    Args:
        pauli     (QubitOperator): Pauli string to exponentiate.
        qubits    (list): List of qubit names.
        parameter (sympy.Symbol, float): Optional parameter to multiple
            the Pauli string.

    Yields:
        cirq.Circuit gates to implement exponent of pauli.

    Raises:
        TypeError: invalid operator for pauli.
        ValueError: if Pauli operator is single-qubit.

    Notes:
        Parameters can be a list of sympy.Symbols or a numerical value.
        The sign of the Pauli operator is kept to multiply the parameter.
        When no parameter passed the coefficient of QubitOperator is used.
    """
    if not isinstance(pauli, QubitOperator):
        raise TypeError('Pauli must be a QubitOperator object.')
    if len(pauli.terms) > 1:
        raise ValueError('QubitOperator with multiple Pauli strings.')
    qbts, paus = zip(*list(pauli.terms.keys())[0])
    if len(qbts) == 1:
        raise ValueError(
            'Pauli operator is single qubit use rotations instead.')
    if parameter is None:
        parameter = list(pauli.terms.values())[0]
    else:
        parameter = numpy.sign(list(pauli.terms.values())[0]) * parameter

    for qbt, pau in zip(qbts, paus):
        yield rot_dic[pau](qubits[qbt], 1.0)

    for j in range(len(qbts) - 1):
        yield cirq.CNOT(qubits[qbts[j]],
                        qubits[qbts[j + 1]])

    yield cirq.rz(parameter)(qubits[qbts[-1]])

    for j in range(len(qbts) - 1, 0, -1):
        yield cirq.CNOT(qubits[qbts[j - 1]],
                        qubits[qbts[j]])

    for qbt, pau in zip(qbts, paus):
        yield rot_dic[pau](qubits[qbt], -1.0)


def trotter_qubitoperator_exponent(
        pauli_operators: QubitOperator,
        qubits: Sequence[cirq.Qid],
        parameters: None) -> List[cirq.Operation]:
    """
    Convert sum of Pauli operators to a circuit by first order T-S.

    This function uses _pauli_exponent_to_circuit to construct the circuit.

    Args:
        pauli_operators (list, QubitOperator): List or QubitOperator of Paulis.
        qubits          (list): List of qubit names.
        parameters       (list): Optional list of parameters.

    Yields:
        cirq.Circuit gates to implement exponent of each Pauli individually.

    Raises:
        TypeError: invalid operator for pauli.
        ValeuError: number of Pauli strings and parameters do not match.

    Notes:
        If the same parameters is used in several Pauli operators a list of
        repeated sympy.Symbols is required.
        Parameters are linked to Pauli operators by order of appearing in the
        list.
        In order to ensure that the order of the parameters is correctly
        attached to the Pauli operator construct the list in parallel.
    """
    if not isinstance(pauli_operators, (list, QubitOperator)):
        raise TypeError('Pauli must be a QubitOperator object.')

    if isinstance(pauli_operators, QubitOperator):
        pauli_operators = list(pauli_operators)

    if parameters:
        if len(pauli_operators) != len(parameters):
            raise ValueError(
                'Number of Pauli strings and parameters must be equal.')

        for pauli, param in zip(pauli_operators, parameters):
            yield pauli_exponent_to_circuit(pauli, qubits, param)
    else:
        for pauli in pauli_operators:
            yield pauli_exponent_to_circuit(pauli, qubits, None)
