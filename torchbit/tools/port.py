class InputPort:
    def __init__(self, wrapper_signal):
        self.wrapper = wrapper_signal

    def get(self):
        if self.wrapper is None:
            return 0
        return self.wrapper.value.integer


class OutputPort:
    def __init__(self, wrapper_signal, set_immediately=False):
        self.wrapper = wrapper_signal
        self.set_immediately = set_immediately

    def set(self, value):
        if self.wrapper is None:
            return
        if self.set_immediately:
            self.wrapper.setimmediatevalue(value)
        else:
            self.wrapper.value = value
