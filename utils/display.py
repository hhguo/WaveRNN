import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import sys


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def stream(message) :
    sys.stdout.write("\r{}".format(message))


def simple_table(item_tuples) :

    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    headings, cells, = [], []

    for item in item_tuples :

        heading, cell = str(item[0]), str(item[1])

        pad_head = True if len(heading) < len(cell) else False

        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]

        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head :
            heading = pad_left + heading + pad_right
        else :
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = '', '', ''

    for i in range(len(item_tuples)) :

        temp_head = '| {} '.format(headings[i])
        temp_body = '| {} '.format(cells[i])

        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1 :
            head += '|'
            body += '|'
            border += '+'

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(' ')


def time_since(started) :
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60 :
        h = int(m // 60)
        m = m % 60
        return '{}h {}m {}s'.format(h, m, s)
    else :
        return '{}m {}s'.format(m, s)


def plot(array) : 
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    ax.xaxis.label.set_color('grey')
    ax.yaxis.label.set_color('grey')
    ax.xaxis.label.set_fontsize(23)
    ax.yaxis.label.set_fontsize(23)
    ax.tick_params(axis='x', colors='grey', labelsize=23)
    ax.tick_params(axis='y', colors='grey', labelsize=23)
    plt.plot(array)


def plot_spec(M) :
    M = np.flip(M, axis=0)
    plt.figure(figsize=(18,4))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.show()

