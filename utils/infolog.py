from datetime import datetime
from threading import Thread
from urllib.request import Request, urlopen

import atexit
import json


_format = '%Y-%m-%d %H:%M:%S.%f'
_file = None
_slack_url = None
_history = []
_cache_size = 2000


def init(filename, slack_url=None):
    global _file, _slack_url
    _close_logfile()
    _file = open(filename, 'a')
    _file.write(
        '\n-----------------------------------------------------------------\n'
    )
    _file.write('Starting new training run\n')
    _file.write(
        '-----------------------------------------------------------------\n')
    _slack_url = slack_url


def log(msg, end='\n', slack=False):
    print(msg, end=end, flush=True)
    global _history
    _history.append('[%s]  %s\n' %
                    (datetime.now().strftime(_format)[:-3], msg))
    if _file is not None and len(_history) % _cache_size == 0:
        _file.write(''.join(_history))
        _history = []
    if slack and _slack_url is not None:
        Thread(target=_send_slack, args=(msg, )).start()


def _close_logfile():
    global _file
    if _file is not None:
        _file.close()
        _file = None


def _send_slack(msg):
    req = Request(_slack_url)
    req.add_header('Content-Type', 'application/json')
    urlopen(
        req,
        json.dumps({
            'username': 'tacotron',
            'icon_emoji': ':taco:',
            'text': '%s' % (msg)
        }).encode())


atexit.register(_close_logfile)
