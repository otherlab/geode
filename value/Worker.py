'''Values evaluated on separate processes via the multiprocessing module'''

from other.core import Prop,listen
import multiprocessing
import errno
import sys

__all__ = ['Worker']

QUIT = 'quit'
NEW_VALUE = 'new value'
SET_VALUE = 'set value'
CREATE_NODE = 'create node'
CREATE_NODE_ACK = 'create node ack'
PULL_NODE = 'pull node'

class ValueProxies(object):
  def __init__(self,conn):
    self.conn = conn
    self.values = {}

  def get(self,name):
    return self.values[name]

  def add(self,name,default):
    '''For compatibility with PropManager.  The value must already exist.'''
    return self.get(name)

  def process(self,tag,data):
    '''Process a message if possible, and return whether we understood it.'''
    if tag==NEW_VALUE:
      name,default = data
      assert name not in self.values
      self.values[name] = Prop(name,default)
    elif tag==SET_VALUE:
      name,value = data
      self.values[name].set(value)
    else:
      return False
    return True

def worker_helper(conn,debug):
  inputs = ValueProxies(conn)
  nodes = {}
  listeners = []
  def process(tag,data):
    if tag==CREATE_NODE:
      name,factory = data
      assert name not in nodes
      node = nodes[name] = factory(inputs)
      if debug:
        print 'worker: send new value %s'%name
      conn.send((NEW_VALUE,(name,None)))
      def push():
        value = None if node.dirty() else node()
        if debug:
          print 'worker: send push %s, %s'%(name,value)
        conn.send((SET_VALUE,(name,value)))
      listeners.append(listen(node,push))
      if debug:
        print 'worker: send create node ack %s'%name
      conn.send((CREATE_NODE_ACK,name))
    elif tag==PULL_NODE:
      name = data
      node = nodes[name]
      node()
    else:
      return inputs.process(tag,data)
    return True
  while 1:
    tag,data = conn.recv()
    if debug:
      print 'worker: recv %s, %s'%(tag,data)
    if tag==QUIT:
      return
    elif not process(tag,data):
      raise ValueError("Unknown tag '%s'"%tag)

class Worker(object):
  def __init__(self,props,debug=False):
    self.debug = debug
    self.conn,child_conn = multiprocessing.Pipe()
    self.worker = multiprocessing.Process(target=worker_helper,args=(child_conn,debug))
    self.outputs = ValueProxies(self.conn)
    self.worker.start()
    self.listeners = []
    for name in props.order:
      self.add_input(name,props.get(name))
    self.crash_timeout = .05
    self.inside_with = False

  def __enter__(self):
    self.inside_with = True
    return self

  def __exit__(self,*args):
    self.inside_with = False
    self.worker.terminate()
    try: # See http://stackoverflow.com/questions/1238349/python-multiprocessing-exit-error
      self.worker.join()
    except OSError,e:
      if e.errno != errno.EINTR:
        raise

  def add_input(self,name,value):
    def changed():
      if self.debug:
        print 'master: send set value %s, %s'%(name,value())
      self.conn.send((SET_VALUE,(name,value())))
    self.listeners.append(listen(value,changed))
    if self.debug:
      print 'master: send new value %s, %s'%(name,value())
    self.conn.send((NEW_VALUE,(name,value())))

  def process(self,timeout=0,count=0):
    '''Check for incoming messages from the worker.'''
    assert self.inside_with
    while self.conn.poll(timeout):
      tag,data = self.conn.recv()
      if self.debug:
        print 'master: recv %s, %s'%(tag,data)
      if not self.outputs.process(tag,data):
        raise ValueError("Unknown tag '%s'"%tag)
      count -= 1
      if not count:
        break

  def create(self,name,factory):
    assert self.inside_with
    '''Create a node on the worker and wait for acknowledgement.'''
    if self.debug:
      print 'master: send create node %s'%name
    self.conn.send((CREATE_NODE,(name,factory)))
    while self.worker.is_alive():
      while self.conn.poll(self.crash_timeout):
        tag,data = self.conn.recv()
        if self.debug:
          print 'master: recv %s, %s'%(tag,data)
        if tag==CREATE_NODE_ACK:
          assert data==name
          return self.outputs.get(name)
        elif not self.outputs.process(tag,data):
          raise ValueError("Unknown tag '%s'"%tag)
    raise RuntimeError("failed to create node '%s'"%name)

  def pull(self,name):
    if self.debug:
      print 'master: send pull node %s'%name
    self.conn.send((PULL_NODE,name))
