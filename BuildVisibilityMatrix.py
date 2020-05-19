import sys

sys.dont_write_bytecode = True


def BuildVisibilityMatrix(Visibility, r_indx, print_enable=False):
    if (print_enable):
        print(Visibility[:, r_indx])

    return Visibility[:, r_indx]
