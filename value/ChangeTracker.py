from other.core import listen

class KeysTracker:
  def __init__(self, keys_value, lookup_value_function):
    self.keys_value = keys_value
    self.lookup_value_function = lookup_value_function
    self.keys_listener = listen(self.keys_value, lambda: self.on_keys_change())
    self.keys_up_to_date = False
    self.tracker = ChangeTracker()

  def on_keys_change(self):
    self.keys_up_to_date = False

  def pull(self):
    if not self.keys_up_to_date:
      new_keys = set(self.keys_value())
      old_keys = set(self.tracker.tracked_items())
      removed_keys = old_keys - new_keys
      added_keys = new_keys - old_keys
      for k in removed_keys:
        self.tracker.untrack(k)
      for k in added_keys:
        self.tracker.track(k, self.lookup_value_function(k))
      self.keys_up_to_date = True

    return self.tracker.pull()

  def refresh(self):
    self.keys_up_to_date = False
    self.tracker.refresh()

class ChangeTracker:
  def __init__(self):
    self._refs = {}
    self._groups = {}
    self._up_to_date = set()
    self._listeners = {}

  def tracked_items(self):
    return self._refs.keys()

  def pull(self):
    result = {}
    for name, ref in self._refs.iteritems():
      if name in self._up_to_date:
        continue
      result[name] = ref()
      self._up_to_date.add(name)
    for name, d in self._groups.iteritems():
      dpull = d.pull()
      if len(dpull): result[name] = dpull

    deleted = [name for name in self._up_to_date if name not in self._refs.keys() and name not in self._groups.keys()]
    for name in deleted:
      result[name] = None

    self._up_to_date -= set(deleted)
    return result

  def refresh(self):
    self._up_to_date.clear()
    for name, d in self._groups.iteritems():
      d.refresh()

  def _on_change(self, name):
    self._up_to_date.discard(name)

  def track(self, name, value_ref):
    self._refs[name] = value_ref
    self._listeners[name] = listen(value_ref, lambda: self._on_change(name))

  def untrack(self, name):
    del self._refs[name]
    del self._listeners[name]

  def track_group(self, name, keys_value, lookup_value_function):
    self._groups[name] = KeysTracker(keys_value, lookup_value_function)

class Synced:
  def refresh(self): pass
  def up_to_date(self): return True
  def pull_changes(self): return None

class SyncedValue(Synced):
  """
    Holds onto a other.Value and tracks if its state changes or refresh is
    called between calls to pull_changes.
  """
  def __init__(self, source_value, vtype):
    self.source_value = source_value
    self._up_to_date = False
    self.vtype = vtype
    self.listener = listen(self.source_value, self.refresh)
  def refresh(self):
    self._up_to_date = False
  def up_to_date(self):
    return self._up_to_date
  def pull_changes(self):
    new_result = self.source_value()
    self._up_to_date = True
    return new_result#[self.vtype, new_result]

class SyncedDict(Synced):
  def __init__(self):
    self._values = {}
    self._deleted = set()

  def set(self, key, value, vtype=""):
    if not isinstance(value, Synced):
      value = SyncedValue(value, vtype)
    self._values[key] = value
    self._deleted.discard(key)
    return self

  def delete(self, key):
    del self._values[key]
    self._deleted.add(key)
    return self

  def refresh(self):
    self._deleted.clear()
    for v in self._values.itervalues(): v.refresh()
    return self

  def keys(self):
    return list(self._values.keys()) + list(self._deleted)

  def up_to_date(self):
    return len(self._deleted) == 0 and \
           all(v.up_to_date() for v in self._values.itervalues())

  def get(self, key):
    return self._values.get(key)

  def pull_changes(self):
    result = {}
    for k in self._deleted: result[k] = None
    self._deleted.clear()
    for k, v in self._values.items():
      if not v.up_to_date():
        result[k] = v.pull_changes()
    return result

class SyncedKeysDict(Synced):
  def __init__(self, keys_value, key_state_to_value_converter, vtype=""):
    self._keys = SyncedValue(keys_value,"")
    self._dict = SyncedDict()
    self._value_converter = key_state_to_value_converter
    self._vtype = vtype

  def _update_keys(self):
    if not self._keys.up_to_date():
      new_keys = set(self._keys.pull_changes())
      old_keys = set(self._dict.keys())
      added_keys = new_keys - old_keys
      removed_keys = old_keys - new_keys
      for k in removed_keys:
        self._dict.delete(k)
      for k in added_keys:
        self._dict.set(k, self._value_converter(k), self._vtype)

  def refresh(self):
    self._keys.refresh()
    self._dict = SyncedDict()
    return self

  def up_to_date(self):
    return self._keys.up_to_date() and self._dict.up_to_date()

  def get(self, key):
    self._update_keys()
    return self._dict.get(key)

  def pull_changes(self):
    self._update_keys()
    return self._dict.pull_changes()

"""
  The Synced classes allow easier differentiation between updates and data when nested
  Here is excerpts of how they might be used:

  server side python script init:
    models = SyncedDict()
    models.set("result", cache(lambda: region_mgr.get_final_geometry()), "geo")
    models.set("error", cache(lambda: region_mgr.get_error_geometry()), "geo")

    sd = SyncedDict()
    sd.set("pick_ids", cache(lambda: [int(id) for id in region_mgr.all_ids()]))
    sd.set("pick_shapes", SyncedKeysDict(picking_ids, get_pick_shape, "geo"))
    sd.set("models", models)

  client side js parsing for generated updates:
    function apply_updates(root, updates) {
      for(var k in updates) {
        var flag = updates[k][0];
        if(flag == "") {
          root[k] = updates[k][1];
        }
        else if(flag == "del") {
          delete root[k];
        }
        else if(flag == "upd") {
          if(!(k in root)) root[k] = {};
          apply_updates(root[k], updates[k][1]);
        }
        else if(flag == "geo") {
          if(!(k in root)) root[k] = new Model().setProgs(programs.tris, programs.color);
          root[k].setData(updates[k][1]);
        }
        else {
          console.log("Unknown flag for \""+k+"\": \""+flag+"\"");
        }
      }
    }

    var state_cache = {};
    function onCommandSuccess (res) {
      apply_updates(state_cache, res[1]);
    }
"""

if __name__ == "__main__":
  from other.core import *

  props = PropManager()
  var_a = props.add("a", 3)
  var_b = props.add("b", 4)
  var_sum = cache(lambda: var_a() + var_b())
  ct = ChangeTracker()
  ct.track('a', var_a)
  ct.track('b', var_b)
  ct.track('sum', var_sum)

  print "updates: %s" % str(ct.pull())
  print "setting 'a' to 5"
  var_a.set(5)
  print "updates: %s" % str(ct.pull())
  print "setting 'b' to 12"
  var_b.set(12)
  print "updates: %s" % str(ct.pull())
  ct.refresh()
  print "refresh: %s" % str(ct.pull())
