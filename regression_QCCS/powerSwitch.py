try:
    import http.client as httpcl
except ImportError:
    import httplib as httpcl

import re
import time

from numpy.matlib import random


class powerSwitch(object):
    def __init__(self, host='powerswitch1.zhinst.com', a_logger=None):
        self.host = host
        self.log = a_logger
        self.switch = powerswitches(host)

    def on(self, a_channel=None):
        if a_channel is None:
            c = list(range(0, 8))
        else:
            c = a_channel if isinstance(a_channel, list) else [a_channel]

        try:
            self.switch.logon()
            # Log in and set channel
            for i in c:
                r_ok = self.switch.onSingle(i)
                if r_ok is False:
                    if self.log is not None:
                        self.log.info('(APC) ON unsuccessful, retry')
                    time.sleep(random.randint(1, 10 + 1))
                else:
                    time.sleep(0.5)
                    if self.log is not None:
                        self.log.info('(APC) ON successful')
            # Log out
            self.switch.logoff()
        except Exception as detail:
            if self.log is not None:
                self.log.warning("%s on %s (power on). RETRY in 3s" % (detail, self.host))
                self.switch.logoff()

    def off(self, a_channel=None):
        if a_channel is None:
            c = list(range(0, 8))
        else:
            c = a_channel if isinstance(a_channel, list) else [a_channel]
        try:
            self.switch.logon()
            for i in c:
                r_ok = False
                while r_ok is False:
                    r_ok = self.switch.offSingle(i)
                    if r_ok is False:
                        if self.log is not None:
                            self.log.info('(APC) OFF unsuccessful, retry')
                        time.sleep(random.randint(1, 10 + 1))
                    else:
                        time.sleep(0.5)
                        if self.log is not None:
                            self.log.info('(APC) OFF successful')
            self.switch.logoff()
        except Exception as detail:
            if self.log is not None:
                self.log.warning("%s on %s (power on)." % (detail, self.host))
                self.switch.logoff()


class powerswitches(object):
    def __init__(self, target='powerswitch1.zhinst.com'):
        m = re.match('^[^0-9]*([0-9]+).*$', target)

        if m:
            self.sid = int(m.group(1)) - 1
        else:
            self.sid = 0
        self.conn = None

    def logon(self):
        attempt = 0
        while (self.conn is None) & (attempt < 10):
            self.conn = httpcl.HTTPConnection('powerswitches.zhinst.com')
            attempt = attempt + 1

    def logoff(self):
        # no action needed
        self.conn = None

    def onSingle(self, channel):
        self.conn.request("GET", "/index.php?ajax=set&sid=%d&chid=%d&val=1&su=1" % (self.sid, channel))
        response = self.conn.getresponse()
        rep = response.read()
        r_ok = (rep.decode() == 'SET:1')

        if not response.status == 200:
            raise RuntimeError("Error %s(%d) during switching on channel %d." % (response.reason, response.status, channel))
        return r_ok

    def offSingle(self, channel):
        self.conn.request("GET", "/index.php?ajax=set&sid=%d&chid=%d&val=0&su=1" % (self.sid, channel))
        response = self.conn.getresponse()
        rep = response.read()
        r_ok = (rep.decode() == 'SET:0')

        if not response.status == 200:
            raise RuntimeError("Error %s(%d) during switching off channel %d." % (response.reason, response.status, channel))
        return r_ok
