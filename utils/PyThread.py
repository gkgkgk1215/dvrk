import threading, time, signal

MILLION = 10**6

class ProgramKilled(Exception):
    pass

def signal_handler(signum, frame):
    raise ProgramKilled

class PyThread(threading.Thread):
    def __init__(self, interval_ms):
        threading.Thread.__init__(self)
        self.interval_ms = interval_ms
        self.cnt = 0.0
        self.nStart = 0.0
        self.nEnd = 0.0

    def start(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self.run, args=(lambda: self.stop_flag,))
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stop_flag = True

    def run(self, stop):
        while True:
            self.nEnd = time.clock()*MILLION    # (us)
            if self.nEnd-self.nStart < self.interval_ms*1000:
                pass
            else:
                # To do
                self.cnt += 1000.0/MILLION*self.interval_ms
                self.nStart = self.nEnd;
            if stop():
                break

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    job = PyThread(10)
    job.start()

    while True:
        try:
            print(job.cnt)
            time.sleep(0.1)
        except ProgramKilled:
            print("Program killed: running cleanup code")
            job.stop()
            break