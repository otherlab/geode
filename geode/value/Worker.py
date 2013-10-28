'''Values evaluated on separate processes via the multiprocessing module'''

from geode import Prop,PropManager,listen,base64_encode,is_windows
import multiprocessing.connection
import multiprocessing
import subprocess
import traceback
import select
import errno
import sys
import os

__all__ = ['Worker','worker_standalone_main']

QUIT = 'quit'
QUIT_ACK = 'quit ack'
NEW_VALUE = 'new value'
SET_VALUE = 'set value'
CREATE_VALUE = 'create value'
PULL_VALUE = 'pull value'
RUN_JOB = 'run job'

class ValueProxies(PropManager):
  def __init__(self,conn):
    self.conn = conn
    self.frozen = True

  def process(self,tag,data):
    '''Process a message if possible, and return whether we understood it.'''
    if tag==NEW_VALUE:
      name,default = data
      if self.contains(name):
        raise RuntimeError("input '%s' already exists on worker process"%name)
      self.frozen = False
      try:
        self.add(name,default)
      finally:
        self.frozen = True
    elif tag==SET_VALUE:
      name,value = data
      self.get(name).set(value)
    else:
      return False
    return True

class Connection(object):
  def __init__(self,side,conn,debug=False):
    self.side = side
    self.conn = conn
    self.debug = debug
    self.inputs = ValueProxies(conn)
    self.outputs = {}
    self.listeners = []
    self.atquits = []

  def add_output(self,name,value):
    assert name not in self.outputs
    self.outputs[name] = value
    if self.debug:
      print '%s: send new value %s'%(self.side,name)
    val = None if value.dirty() else value()
    self.conn.send((NEW_VALUE,(name,val)))
    def push():
      val = None if value.dirty() else value()
      if self.debug:
        print '%s: send push %s, %s'%(self.side,name,val)
      self.conn.send((SET_VALUE,(name,val)))
    self.listeners.append(listen(value,push))

  def atquit(self,f):
    '''Run the given function before a normal quit'''
    self.atquits.append(f)

  def process(self,timeout=0,count=0):
    '''Check for incoming messages from the master.'''
    while self.conn.poll(timeout):
      tag,data = self.conn.recv()
      if self.debug:
        print '%s: recv %s, %s'%(self.side,tag,data)
      if tag==QUIT:
        for f in self.atquits:
          f()
        self.conn.send((QUIT_ACK,()))
        sys.exit()
      elif tag==CREATE_VALUE:
        name,factory = data
        value = factory(self.inputs)
        self.add_output(name,value)
      elif tag==PULL_VALUE:
        name = data
        node = self.outputs[name]
        node()
      elif tag==RUN_JOB:
        # Execute function with connection and given extra arguments
        f,args,kwargs = data
        f(self,*args,**kwargs)
      elif not self.inputs.process(tag,data):
        raise ValueError("Unknown tag '%s'"%tag)
      count -= 1
      if not count:
        break

def worker_main(conn,debug):
  conn = Connection('worker',conn,debug=debug)
  conn.process(timeout=None)

def worker_standalone_main(key):
  debug,port,authkey = key.split(',')
  debug = debug=='1'
  port = int(port)
  conn = multiprocessing.connection.Client(('localhost',port),authkey=authkey)
  worker_main(conn,debug)

class QuitAck(BaseException):
  pass

class Worker(object):
  def __init__(self,debug=False,quit_timeout=1.,command=None,accept_timeout=2.):
    '''Create a new worker process with two way communication via Value objects.
    The worker blocks until told to launch values or jobs.

    If command is specified, the worker is created using subprocess.Popen(command+[key])
    and connected to via Listener/Client rather than the normal multiprocessing.Process
    mechanism.  This avoids strange problems with gdb and mpi on Mac.  The worker should
    immediately call worker_standalone_main.'''
    self.debug = debug
    self.inside_with = False
    self.command = command
    if command is None:
      self.conn,child_conn = multiprocessing.Pipe()
      worker = multiprocessing.Process(target=worker_main,args=(child_conn,debug))
      worker.start()
      self.worker_is_alive = worker.is_alive
      self.worker_join = worker.join
      self.worker_terminate = worker.terminate
    else:
      key = base64_encode(os.urandom(32))
      listener = multiprocessing.connection.Listener(('localhost',0),authkey=key)
      worker = self.worker = subprocess.Popen(command+['%d,%d,%s'%(bool(debug),listener.address[1],key)],shell=is_windows()) # shell=True is required on Windows, but doesn't work on Mac
      # Ugly hack to avoid blocking forever if client never shows up.
      # Borrowed from http://stackoverflow.com/questions/357656/proper-way-of-cancelling-accept-and-closing-a-python-processing-multiprocessing
      listener.fileno = lambda:listener._listener._socket.fileno()
      r,w,e = select.select((listener,),(),(),accept_timeout)
      if not r:
        raise RuntimeError('worker process failed to launch')
      self.conn = listener.accept()
      self.worker_is_alive = lambda:worker.poll() is None
      self.worker_terminate = worker.terminate
      self.worker_join = lambda timeout:None
    self.outputs = ValueProxies(self.conn)
    self.listeners = []
    self.crashed = False
    self.quit_timeout = quit_timeout

  def __enter__(self):
    self.inside_with = True
    return self

  def __exit__(self,*args):
    def safe_join(timeout):
      # See http://stackoverflow.com/questions/1238349/python-multiprocessing-exit-error for why we need this try block.
      try:
        self.worker_join(timeout)
      except OSError,e:
        if e.errno != errno.EINTR:
          raise
    if self.debug:
      print 'master: send quit'
    # First try telling the process to quit peacefully
    self.conn.send((QUIT,()))
    # Wait a little while for a quit acknowledgement.  This is necessary for clean shutdown.
    try:
      self.process(timeout=self.quit_timeout)
    except QuitAck:
      pass
    # Attempt to join with the child process peacefully
    safe_join(self.quit_timeout)
    # If the peaceful method doesn't work, use force
    self.worker_terminate()
    safe_join(None)

  def add_props(self,props):
    for name in props.order:
      self.add_input(name,props.get(name))

  def add_input(self,name,value):
    assert self.inside_with
    if self.debug:
      print 'master: send new value %s, %s'%(name,value())
    self.conn.send((NEW_VALUE,(name,value())))
    def changed():
      if self.debug:
        print 'master: send set value %s, %s'%(name,value())
      self.conn.send((SET_VALUE,(name,value())))
    self.listeners.append(listen(value,changed))

  def process(self,timeout=0,count=0):
    '''Check for incoming messages from the worker.'''
    assert self.inside_with
    remaining = 1e10 if timeout is None else timeout
    while 1:
      # Poll without hanging if the worker process dies
      if remaining<0:
        return # Ran out of time
      if not self.worker_is_alive():
        self.crashed = True
        raise IOError('worker process crashed')
      # Don't wait for too long in order to detect crashes
      pause = min(remaining,.21)
      if self.conn.poll(pause):
        # Process message
        tag,data = self.conn.recv()
        if self.debug:
          print 'master: recv %s, %s'%(tag,data)
        if tag==QUIT_ACK:
          raise QuitAck()
        if not self.outputs.process(tag,data):
          raise ValueError("Unknown tag '%s'"%tag)
        count -= 1
        if not count:
          return # Hit message limit
      elif not pause:
        break
      remaining -= pause

  def wait_for_output(self,name):
    while not self.outputs.contains(name):
      self.process(timeout=None,count=1)
    return self.outputs.get(name)

  def create(self,name,factory):
    '''Create a node on the worker and wait for acknowledgement.'''
    assert self.inside_with
    if self.debug:
      print 'master: send create node %s'%name
    self.conn.send((CREATE_VALUE,(name,factory)))
    return self.wait_for_output(name)

  def pull(self,name):
    if self.debug:
      print 'master: send pull node %s'%name
    self.conn.send((PULL_VALUE,name))

  def run(self,f,*args,**kwargs):
    '''Execute f(conn,*args,**kwargs) on the worker process.
    If f is long running, it should periodically call conn.process(...).'''
    if self.debug:
      print 'master: send run job %s'%f
    self.conn.send((RUN_JOB,(f,args,kwargs)))
