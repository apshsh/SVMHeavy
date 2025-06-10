def convgen(h):
    import pyheavy
    if type(h) is None:
        pyheavy.compush("none")
    elif type(h) is int:
        pyheavy.compush("int")
        pyheavy.intpush(h)
    elif type(h) is float:
        pyheavy.compush("float")
        pyheavy.dblpush(h)
    elif type(h) is complex:
        pyheavy.compush("complex")
        pyheavy.dblpush(h.real)
        pyheavy.dblpush(h.imag)
    elif type(h) is str:
        pyheavy.compush("string")
        pyheavy.strpush(h)
    elif type(h) is list:
        pyheavy.compush("list")
        pyheavy.intpush(len(h))
        for x in h:
            convgen(x)
    elif type(h) is tuple:
        pyheavy.compush("tuple")
        pyheavy.intpush(len(h))
        for x in h:
            convgen(x)
    else:
        pyheavy.compush("badtype")

