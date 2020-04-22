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

import unittest

from openfermion import QubitOperator

from openfermioncirq.utils import (
    qubitoperator_to_pauli_string, qubitoperator_to_pauli_sum)


class QubitOperatorTranslatorTest(unittest.TestCase):
    """Test class for qubitop translator."""

    def test_function_raises(self):
        """Test function raises."""
        qop = QubitOperator('X0 Y1', 1.0) + QubitOperator('Z0 Z1', -0.5)
        with self.assertRaises(TypeError):
            qubitoperator_to_pauli_string(1.0)
        with self.assertRaises(ValueError):
            qubitoperator_to_pauli_string(qop)
        with self.assertRaises(TypeError):
            qubitoperator_to_pauli_sum([5.0])
