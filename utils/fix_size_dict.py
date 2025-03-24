from collections import OrderedDict


class FixedSizeOrderedDict(OrderedDict):
    def __init__(self, max_len, *args, **kwargs):
        self.max_len = max_len
        super(FixedSizeOrderedDict, self).__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        if len(self) >= self.max_len:
            self.popitem(last=False)  # 移除第一个添加的元素
        super(FixedSizeOrderedDict, self).__setitem__(key, value)
