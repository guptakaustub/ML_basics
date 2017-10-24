from numpy import genfromtxt

data = genfromtxt("data.csv", delimiter=",")
N = float(len(data))

def get_error(m, b):
    error = 0
    for points in data:
        x = points[0]
        y = points[1]
        error = error + (y - ((m * x) + b)) ** 2
    return error/N

def gradient_descent(m_cur, b_cur, learningRate):
    m_sum = 0
    b_sum = 0

    # Getting Differential
    for points in data:
        x = points[0]
        y = points[1]

        m_sum += (((m_cur * x) + b_cur) - y) * x
        b_sum += (((m_cur * x) + b_cur) - y)

    m_sum = m_sum * (2 / N)
    b_sum = b_sum * (2 / N)

    new_m = m_cur - (learningRate * m_sum)
    new_b = b_cur - (learningRate * b_sum)

    return new_m, new_b

def learning_iterations(number):
    m_learn = 0
    b_learn = 0
    for i in range(number):
        m_learn, b_learn = gradient_descent(m_learn, b_learn, 0.0001)
    return m_learn, b_learn

ei = get_error(0,0)
n_m, n_b = learning_iterations(500)
print "After 500 Iterations"
print "New Value of Slope: {0}\nNew Value of Intercept: {1}".format(n_m, n_b)
ef = get_error(n_m, n_b)
print "Initial Error: {0}\nFinal Error: {1}".format(ei, ef)
print "Score: {0}".format(1 - ef/ei)