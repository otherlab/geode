#include <other/core/value/Action.h>
#include <other/core/utility/const_cast.h>
#include <iostream>
namespace other{

using std::cout;
using std::endl;

const Action* Action::current = 0;

Action::Action()
    :inputs_(0),executing(false) {}

Action::~Action() {
    clear_dependencies();
}

int Action::inputs() const {
    int count = 0;
    for (ValueBase::Link* link=inputs_; link; link=link->action_next)
        count++;
    return count;
}

void Action::clear_dependencies() const {
    ValueBase::Link* link = inputs_;
    while (link) {
        ValueBase::Link* next = link->action_next;
        // Remove from the value's linked list
        *link->value_prev = link->value_next;
        if (link->value_next)
            link->value_next->value_prev = link->value_prev;
        delete link;
        link = next;
    }
    inputs_ = 0;
}

void Action::depend_on(const ValueBase& value) const {
    // Create a new action
    ValueBase::Link* link = new ValueBase::Link;
    link->value = &value;
    link->action = const_cast_(this);

    // Insert it into our linked list
    link->action_next = inputs_;
    link->action_prev = &inputs_;
    if (inputs_)
        inputs_->action_prev = &link->action_next;
    inputs_ = link;

    // Insert it into the value's linked list
    link->value_next = value.actions;
    link->value_prev = &value.actions;
    if (value.actions)
        value.actions->value_prev = &link->value_next;
    value.actions = link;
}

static bool ignored_executing = false;

Action::Executing::Executing()
    : executing(ignored_executing), parent(Action::current) {
    Action::current = 0;
}

Action::Executing::Executing(const Action& self)
    : executing(self.executing), parent(Action::current) {
    if (executing)
        throw RuntimeError("cyclic dependency detected in Value computation");
    self.clear_dependencies();
    Action::current = &self;
    executing = true;
}

void Action::dump_dependencies(int indent) const {
  for (ValueBase::Link* link=inputs_; link; link=link->action_next)
    link->value->dump(indent+1);
}

std::vector<Ptr<const ValueBase> > Action::get_dependencies() const {
  std::vector<Ptr<const ValueBase> > result;
  for (ValueBase::Link* link=inputs_; link; link=link->action_next) {
    result.push_back(ptr(link->value));
  }
  return result;
}

}
