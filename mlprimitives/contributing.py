import inspect
import json
import os
import warnings

from mlprimitives import MLPRIMITIVES_JSONS_PATH, load_primitive
from mlprimitives.utils import import_object


class ContributingHelper:

    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.contributor = '{} <{}>'.format(self.name, self.email)

    def _create_json(self, primitive_name, primitive_type, primitive_subtype,
                     args, output, fixed, tunable, description):

        primitive_dict = {
            'name': primitive_name,
            'contributors': [self.contributor],
            'description': description,
            'classifiers': {
                'type': primitive_type,
                'subtype': primitive_subtype
            },
            'primitive': primitive_name,
            'produce': {
                'args': args,
                'output': output
            },
            'hyperparameters': {
                'fixed': fixed,
                'tunable': tunable
            }
        }

        # base, name = primitive_name.rsplit('.', 1)
        # primitive_dir = os.path.join(MLPRIMITIVES_JSONS_PATH, *base.split('.'))
        primitive_path = os.path.join(MLPRIMITIVES_JSONS_PATH, primitive_name + '.json')

        # if not os.path.exists(primitive_dir):
        #     os.makedirs(primitive_dir)
        # elif not os.path.isdir(primitive_dir):
        #     raise ValueError('{} already exists and is not a folder.'.format(primitive_dir))
        # if not os.path.isdir(primitive_path):
        #     raise ValueError('Primitive {}.json already exists.'.format(primitive_path))

        with open(primitive_path, 'w') as json_file:
            json.dump(primitive_dict, json_file, indent=4)

    def _validate_callable(self, primitive, arguments):
        signature = inspect.signature(primitive)
        parameters = signature.parameters.copy()
        for argument in arguments:
            parameter = parameters.pop(argument, None)
            if not parameter:
                raise ValueError('Invalid argument: {}'.format(argument))

        for parameter in parameters.values():
            if parameter.name != 'self':
                if parameter.default is inspect.Signature.empty:
                    raise ValueError('Unspecified argument: {}'.format(parameter.name))
                else:
                    warning = 'Unspecified argument {}. Default value will be used'
                    warnings.warn(warning.format(parameter.name), SyntaxWarning)

    def _get_argument_names(self, args, fixed=None, tunable=None):
        names = list()
        for argument in args:
            keyword = argument.get('keyword')
            if keyword:
                names.append(keyword)
            else:
                names.append(argument['name'])

        if fixed:
            names.extend(fixed.keys())
        if tunable:
            names.extend(tunable.keys())

        return names

    def add_function_primitive(self, primitive, primitive_type, primitive_subtype,
                               args, output, fixed, tunable, description):
        if isinstance(primitive, str):
            primitive_name = primitive
            try:
                primitive = import_object(primitive_name)
            except ImportError:
                raise ValueError('Invalid primitive name: {}'.format(primitive_name)) from None

        if inspect.isclass(primitive):
            raise ValueError('Primitive is a Class. Please use `add_class_primitive`.')
        # elif inspect.isfunction(primitive):
        #     primitive_name = self._create_primitive(primitive, category)

        arguments = self._get_argument_names(args, fixed, tunable)
        self._validate_callable(primitive, arguments)

        self._create_json(primitive_name, primitive_type, primitive_subtype,
                          args, output, fixed, tunable, description)

        return load_primitive(primitive_name)

    def add_primitive(self, primitive, primitive_type, primitive_subtype, fit, fit_args,
                      produce, produce_args, output, fixed, tunable, description):
        if isinstance(primitive, str):
            primitive_name = primitive
            try:
                primitive = import_object(primitive_name)
            except ImportError:
                raise ValueError('Invalid primitive name: {}'.format(primitive_name)) from None

        if inspect.isfunction(primitive):
            raise ValueError('Primitive is a function. Please use `add_function_primitive`.')
        # elif inspect.isclass(primitive):
        #     primitive_name = self._create_primitive(primitive, category)

        hyperparams = list(fixed.keys()) + list(tunable.keys())
        self._validate_callable(primitive, hyperparams)

        if fit is not None:
            fit_arguments = self._get_argument_names(fit_args)
            fit_method = getattr(primitive, fit, None)
            if fit_method is None:
                raise ValueError('Invalid fit method: {}'.format(fit))

            self._validate_callable(fit_method, fit_arguments)

        produce_arguments = self._get_argument_names(produce_args)
        produce_method = getattr(primitive, produce, None)
        if produce_method is None:
            raise ValueError('Invalid produce method: {}'.format(produce))

        self._validate_callable(produce_method, produce_arguments)

        self._create_json(primitive_name, primitive_type, primitive_subtype,
                          args, output, fixed, tunable, description)

        return load_primitive(primitive_name)
