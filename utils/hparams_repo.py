from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import re
import six


PARAM_RE = re.compile(r"""
    (?P<name>[a-zA-Z][\w\.]*)      # variable name: "var" or "x"
    (\[\s*(?P<index>\d+)\s*\])?  # (optional) index: "1" or None
    \s*=\s*
    ((?P<val>[^,\[]*)            # single value: "a" or None
    |
    \[(?P<vals>[^\]]*)\])       # list of values: None or "1,2,3"
    ($|,\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
    """Helper function for raising a value error for bad assignment."""
    raise ValueError(
          'Could not parse hparam \'%s\' of type \'%s\' with value \'%s\' in %s' %
          (name, var_type.__name__, value, values))


def _reuse_fail(name, values):
    """Helper function for raising a value error for reuse of name."""
    raise ValueError('Multiple assignments to variable \'%s\' in %s' %
          (name, values))


def _process_scalar_value(name, parse_fn, var_type, m_dict, values,
                          results_dictionary):
    """Update results_dictionary with a scalar value.

    Used to update the results_dictionary to be returned by parse_values when
    encountering a clause with a scalar RHS (e.g.  "s=5" or "arr[0]=5".)

    Mutates results_dictionary.

    Args:
        name: Name of variable in assignment ("s" or "arr").
        parse_fn: Function for parsing the actual value.
        var_type: Type of named variable.
        m_dict: Dictionary constructed from regex parsing.
        m_dict['val']: RHS value (scalar)
        m_dict['index']: List index value (or None)
        values: Full expression being parsed
        results_dictionary: The dictionary being updated for return by the parsing
        function.

    Raises:
        ValueError: If the name has already been used.
    """
    try:
        parsed_value = parse_fn(m_dict['val'])
    except ValueError:
        _parse_fail(name, var_type, m_dict['val'], values)

    # If no index is provided
    if not m_dict['index']:
        if name in results_dictionary:
            _reuse_fail(name, values)
        results_dictionary[name] = parsed_value
    else:
        if name in results_dictionary:
        # The name has already been used as a scalar, then it
        # will be in this dictionary and map to a non-dictionary.
            if not isinstance(results_dictionary.get(name), dict):
                _reuse_fail(name, values)
        else:
            results_dictionary[name] = {}

        index = int(m_dict['index'])
        # Make sure the index position hasn't already been assigned a value.
        if index in results_dictionary[name]:
            _reuse_fail('{}[{}]'.format(name, index), values)
        results_dictionary[name][index] = parsed_value


def _process_list_value(name, parse_fn, var_type, m_dict, values,
                        results_dictionary):
    """Update results_dictionary from a list of values.

    Used to update results_dictionary to be returned by parse_values when
    encountering a clause with a list RHS (e.g.  "arr=[1,2,3]".)

    Mutates results_dictionary.

    Args:
        name: Name of variable in assignment ("arr").
        parse_fn: Function for parsing individual values.
        var_type: Type of named variable.
        m_dict: Dictionary constructed from regex parsing.
        m_dict['val']: RHS value (scalar)
        values: Full expression being parsed
        results_dictionary: The dictionary being updated for return by the parsing
        function.

    Raises:
        ValueError: If the name has an index or the values cannot be parsed.
    """
    if m_dict['index'] is not None:
        raise ValueError('Assignment of a list to a list index.')
    elements = filter(None, re.split('[ ,]', m_dict['vals']))
    # Make sure the name hasn't already been assigned a value
    if name in results_dictionary:
        raise _reuse_fail(name, values)
    try:
        results_dictionary[name] = [parse_fn(e) for e in elements]
    except ValueError:
        _parse_fail(name, var_type, m_dict['vals'], values)


def _cast_to_type_if_compatible(name, param_type, value):
    """Cast hparam to the provided type, if compatible.

    Args:
        name: Name of the hparam to be cast.
        param_type: The type of the hparam.
        value: The value to be cast, if compatible.

    Returns:
        The result of casting `value` to `param_type`.

    Raises:
        ValueError: If the type of `value` is not compatible with param_type.
        * If `param_type` is a string type, but `value` is not.
        * If `param_type` is a boolean, but `value` is not, or vice versa.
        * If `param_type` is an integer type, but `value` is not.
        * If `param_type` is a float type, but `value` is not a numeric type.
    """
    fail_msg = (
        "Could not cast hparam '%s' of type '%s' from value %r" %
        (name, param_type, value))

    # If `value` is already of type `param_type`, return it directly.
    # `isinstance` is too weak (e.g. isinstance(True, int) == True).
    if type(value) == param_type:  # pylint: disable=unidiomatic-typecheck
        return value

    # Some callers use None, for which we can't do any casting/checking. :(
    if issubclass(param_type, type(None)):
        return value

    # Avoid converting a non-string type to a string.
    if (issubclass(param_type, (six.string_types, six.binary_type)) and
        not isinstance(value, (six.string_types, six.binary_type))):
        raise ValueError(fail_msg)

    # Avoid converting a number or string type to a boolean or vice versa.
    if issubclass(param_type, bool) != isinstance(value, bool):
        raise ValueError(fail_msg)

    # Avoid converting float to an integer (the reverse is fine).
    if (issubclass(param_type, numbers.Integral) and
        not isinstance(value, numbers.Integral)):
        raise ValueError(fail_msg)

    # Avoid converting a non-numeric type to a numeric type.
    if (issubclass(param_type, numbers.Number) and
        not isinstance(value, numbers.Number)):
        raise ValueError(fail_msg)

    return param_type(value)


def parse_values(values, type_map, ignore_unknown=False):
    """Parses hyperparameter values from a string into a python map.

    `values` is a string containing comma-separated `name=value` pairs.
    For each pair, the value of the hyperparameter named `name` is set to
    `value`.

    If a hyperparameter name appears multiple times in `values`, a ValueError
    is raised (e.g. 'a=1,a=2', 'a[1]=1,a[1]=2').

    If a hyperparameter name in both an index assignment and scalar assignment,
    a ValueError is raised.  (e.g. 'a=[1,2,3],a[0] = 1').

    The hyperparameter name may contain '.' symbols, which will result in an
    attribute name that is only accessible through the getattr and setattr
    functions.  (And must be first explicit added through add_hparam.)

    WARNING: Use of '.' in your variable names is allowed, but is not well
    supported and not recommended.

    The `value` in `name=value` must follows the syntax according to the
    type of the parameter:

    *  Scalar integer: A Python-parsable integer point value.  E.g.: 1,
        100, -12.
    *  Scalar float: A Python-parsable floating point value.  E.g.: 1.0,
        -.54e89.
    *  Boolean: Either true or false.
    *  Scalar string: A non-empty sequence of characters, excluding comma,
        spaces, and square brackets.  E.g.: foo, bar_1.
    *  List: A comma separated list of scalar values of the parameter type
        enclosed in square brackets.  E.g.: [1,2,3], [1.0,1e-12], [high,low].

    When index assignment is used, the corresponding type_map key should be the
    list name.  E.g. for "arr[1]=0" the type_map must have the key "arr" (not
    "arr[1]").

    Args:
        values: String.  Comma separated list of `name=value` pairs where
        'value' must follow the syntax described above.
        type_map: A dictionary mapping hyperparameter names to types.  Note every
        parameter name in values must be a key in type_map.  The values must
        conform to the types indicated, where a value V is said to conform to a
        type T if either V has type T, or V is a list of elements of type T.
        Hence, for a multidimensional parameter 'x' taking float values,
        'x=[0.1,0.2]' will parse successfully if type_map['x'] = float.
        ignore_unknown: Bool. Whether values that are missing a type in type_map
        should be ignored. If set to True, a ValueError will not be raised for
        unknown hyperparameter type.

    Returns:
        A python map mapping each name to either:
        * A scalar value.
        * A list of scalar values.
        * A dictionary mapping index numbers to scalar values.
        (e.g. "x=5,L=[1,2],arr[1]=3" results in {'x':5,'L':[1,2],'arr':{1:3}}")

    Raises:
        ValueError: If there is a problem with input.
        * If `values` cannot be parsed.
        * If a list is assigned to a list index (e.g. 'a[1] = [1,2,3]').
        * If the same rvalue is assigned two different values (e.g. 'a=1,a=2',
        'a[1]=1,a[1]=2', or 'a=1,a=[1]')
    """
    results_dictionary = {}
    pos = 0
    while pos < len(values):
        m = PARAM_RE.match(values, pos)
        if not m:
            raise ValueError('Malformed hyperparameter value: %s' % values[pos:])
        # Check that there is a comma between parameters and move past it.
        pos = m.end()
        # Parse the values.
        m_dict = m.groupdict()
        name = m_dict['name']
        if name not in type_map:
            if ignore_unknown:
                continue
            raise ValueError('Unknown hyperparameter type for %s' % name)
        type_ = type_map[name]

        # Set up correct parsing function (depending on whether type_ is a bool)
        if type_ == bool:
            def parse_bool(value):
                if value in ['true', 'True']:
                    return True
                elif value in ['false', 'False']:
                    return False
                else:
                    try:
                        return bool(int(value))
                    except ValueError:
                        _parse_fail(name, type_, value, values)
            parse = parse_bool
        else:
            parse = type_

        # If a singe value is provided
        if m_dict['val'] is not None:
            _process_scalar_value(name, parse, type_, m_dict, values,
                                  results_dictionary)

        # If the assigned value is a list:
        elif m_dict['vals'] is not None:
            _process_list_value(name, parse, type_, m_dict, values,
                                results_dictionary)

        else:  # Not assigned a list or value
            _parse_fail(name, type_, '', values)

    return results_dictionary


class HParamsRepo(object):

    def __init__(self, **kwargs):
        self._hparam_types = {}
        for name, value in six.iteritems(kwargs):
            self.add_hparam(name, value)

    def __add__(self, other):
        a_values = self.hparams_to_dict()
        b_values = other.hparams_to_dict()
        c_hparams = Configuration()
        for name in a_values:
            c_hparams.add_hparam(name, a_values[name])
        for name in b_values:
            c_hparams.add_hparam(name, b_values[name])
        return c_hparams

    def add_hparam(self, name, value):
        if getattr(self, name, None) is not None:
            raise ValueError('Hyperparameter name is reserved: %s' % name)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    'Multi-valued hyperparameters cannot be empty: %s' % name)
            self._hparam_types[name] = (type(value[0]), True)
        else:
            self._hparam_types[name] = (type(value), False)
        setattr(self, name, value)

    def set_hparam(self, name, value):
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
            if not is_list:
                raise ValueError(
                    'Must not pass a list for single-valued parameter: %s' % name)
            setattr(self, name, [
                _cast_to_type_if_compatible(name, param_type, v) for v in value])
        else:
            if is_list:
                raise ValueError(
                    'Must pass a list for multi-valued parameter: %s.' % name)
            setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

    def parse(self, values):
        type_map = {}
        for name, t in self._hparam_types.items():
            param_type, _ = t
            type_map[name] = param_type

        values_map = parse_values(values, type_map)
        return self.override_from_dict(values_map)

    def override_from_dict(self, values_dict):
        for name, value in values_dict.items():
            self.set_hparam(name, value)
        return self

    def hparams_to_dict(self):
        values = {n: getattr(self, n) for n in self._hparam_types.keys()}
        return values

    def hparams_debug_string(self):
        values = {n: getattr(self, n) for n in self._hparam_types.keys()}
        hparams = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
        return 'Hyperparameters:\n' + '\n'.join(hparams)
