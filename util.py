
def zigzag(n):
    indexorder = sorted(((x,y) for x in range(n) for y in range(n)),
                        key=lambda (x, y): (x+y, -y if (x+y) % 2 else y))
    return indexorder


def triangle(n):
    tri_matrix = [[1 for i in range(n-j-1)] + [0 for i in range(j+1)] for j in range(n)]
    return tri_matrix
