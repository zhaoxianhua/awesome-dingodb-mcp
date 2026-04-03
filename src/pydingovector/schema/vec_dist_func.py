from sqlalchemy import FunctionElement, Float


class l2_distance(FunctionElement):
    """Vector distance function: l2_distance.

    Attributes:
    type : result type
    """
    type = Float()

    def __init__(self, *args):
        super().__init__()
        self.args = args