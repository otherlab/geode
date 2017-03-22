// Implementation of template methods for RankedTree
// This should only be included from RankedTree.h
#pragma once
namespace geode {

template<class T> auto RankedTree<T>::debug_print(std::ostream& os, const char prefix, const int indent, const Node* node) const -> void {
  for(int i = 0; i < indent; ++i) os << ' ';
  os << prefix << ' ';
  if(node) {
    os << (node->is_red() ? "RED   " : "BLACK ") << node->data << " @ " << node << " of rank " << node->rank() << '\n';
    debug_print(os, 'L', indent + 2, node->left.get());
    debug_print(os, 'R', indent + 2, node->right.get());
  }
  else {
    os << "BLACK leaf @ " << node << '\n';
  }
}

template<class T> auto RankedTree<T>::Node::test_local_invarients() const -> void {
  GEODE_ASSERT(parent != this); // Parent should never point back to self (Nor should cycles exist anywhere, but this an easy one to check)
  GEODE_ASSERT(parent || is_black()); // Root node is always black
  GEODE_ASSERT(!parent || parent->left.get() == this || parent->right.get() == this); // If parent is set, this should be one of the children
  GEODE_ASSERT(!left || left->parent == this); // Children should point back to self
  GEODE_ASSERT(!right || right->parent == this); // Children should point back to self
  GEODE_ASSERT(is_black() || (is_black(parent) && is_black(left.get()) && is_black(right.get()))); // No consecutive red nodes
  GEODE_ASSERT(left ? true : !right); // Node should always be left leaning
  GEODE_ASSERT(!(is_red(left.get()) && is_red(right.get()))); // Should never have 4 nodes
  GEODE_ASSERT(rank() >= 1 + (left ? left->rank() : 0));
}


template<class T> auto RankedTree<T>::Node::side() const -> Side {
  assert(parent); // Should probably not be calling this function on the root
  if(parent) {
    assert((this == parent->left) != (this == parent->right));
    return (this == parent->left) ? Side::LEFT : Side::RIGHT;
  }
  else {
    return Side::LEFT;
  }
}

template<class T> auto RankedTree<T>::Node::next() -> Node* {
  // Next node is min below right if it exists
  if(right) return &right->min_child();
  // Walk upwards until we find a node with a parent greater than where we started
  for(Node* iter = this; iter->parent; iter = iter->parent) {
    if(iter->is_left())
      return iter->parent;
  }
  return nullptr; // If every ancestor was less current node is max
}

template<class T> auto RankedTree<T>::Node::prev() -> Node* {
  // Logic is as Node::next with directions reversed
  if(left) return &left->max_child();
  for(Node* iter = this; iter->parent; iter = iter->parent) {
    if(iter->is_right())
      return iter->parent;
  }
  return nullptr;
}

template<class T> auto RankedTree<T>::parent_link(Node& node) -> NodeLink& {
  if(node.parent) {
    return node.is_left() ? node.parent->left : node.parent->right;
  }
  else {
    assert(m_root.get() == &node);
    return m_root; // Link to root node is m_root
  }
}

template<class T> auto RankedTree<T>::test_subtree_invarients(const Node* node) const -> SubtreeStats {
  SubtreeStats result;
  if(!node) {
    result.total_size = 0;
    result.black_height = 1;
    result.max_height = 1;
  }
  else {
    // Only the root node should have no parent
    GEODE_ASSERT((node->parent == nullptr) == (node == m_root.get()));
    node->test_local_invarients();

    const auto l_stats = test_subtree_invarients(node->left.get());
    const auto r_stats = test_subtree_invarients(node->right.get());

    GEODE_ASSERT(node->rank() == l_stats.total_size + 1);
    result.total_size = l_stats.total_size + r_stats.total_size + 1;
    GEODE_ASSERT(l_stats.black_height == r_stats.black_height);
    result.black_height = l_stats.black_height; // Use left value since they are the same
    if(node->is_black()) {
      result.black_height += 1;
    }
    result.max_height = 1 + max(l_stats.max_height, r_stats.max_height);
    GEODE_ASSERT(2*result.black_height >= result.max_height); // Should have at least every other link black
    GEODE_ASSERT(result.total_size <= 1<<result.max_height);
  }
  return result;  
}

template<class T> auto RankedTree<T>::test_global_invarients() const -> void
{ test_subtree_invarients(m_root.get()); }

// This shouldn't alter any of the properties of the graph
template<class T> auto RankedTree<T>::swap_positions(Node& n0, Node& n1) -> void {
  assert(&n0 != &n1);
  if(&n0 == &n1) return;
  std::swap(parent_link(n0), parent_link(n1));
  std::swap(n0.left,    n1.left);
  std::swap(n0.right,   n1.right);
  std::swap(n0.parent,  n1.parent);
  std::swap(n0.m_rank_and_color,  n1.m_rank_and_color);
  // Fix parent points on children
  for(Node* n : {&n0, &n1}) {
    assert(n->left || !n->right); // Check node is left leaning so we don't need to check right link
    if(n->left) {
      assert(n->left->parent != n); // Should be pointing to other node we are swapping with
      n->left->parent = n;
      if(n->right) {
        assert(n->right->parent != n); // Should be pointing to other node we are swapping with
        n->right->parent = n;
      }
    }
  }
}

// Swaps h and h->right adjusting colors and other pointers accordingly
template<class T> auto RankedTree<T>::rotate_left(NodeLink& h) -> void {
  assert(h && h->right);
  h->right->set_color(h->color());
  h->set_color(Color::RED);
  auto& orig_h = *h;
  auto& orig_x = *(h->right);
  swap(h, orig_h.right);
  swap(orig_x.left, orig_h.right);
  orig_x.parent = orig_h.parent;
  orig_h.parent = &orig_x;
  if(orig_h.right) {
    assert(orig_h.right->parent == &orig_x);
    orig_h.right->parent = &orig_h;
  }
  orig_x.update_rank(orig_h.rank());
  assert(h.get() == &orig_x);
  assert(h->left.get() == &orig_h);
}

// Swaps h and h->right adjusting colors and other pointers accordingly
template<class T> auto RankedTree<T>::rotate_right(NodeLink& h) -> void {
  assert(h && h->left);
  h->left->set_color(h->color());
  h->set_color(Color::RED);
  auto& orig_h = *h;
  auto& orig_x = *(h->left);
  swap(h, orig_h.left);
  swap(orig_x.right, orig_h.left);
  orig_x.parent = orig_h.parent;
  orig_h.parent = &orig_x;
  if(orig_h.left) {
    assert(orig_h.left->parent == &orig_x);
    orig_h.left->parent = &orig_h;
  }
  orig_h.update_rank(-orig_x.rank());
  assert(h.get() == &orig_x);
  assert(h->right.get() == &orig_h);
}

template<class T> auto RankedTree<T>::flip_color(Node& node) -> void {
  assert(node.left && node.right);
  assert(node.left->color() == node.right->color());
  node.m_rank_and_color ^= 1;
  node.left->m_rank_and_color ^= 1;
  node.right->m_rank_and_color ^= 1;
}

template<class T> auto RankedTree<T>::fixup(NodeLink& h) -> bool {
  bool changed = false;
  // Fix any newly created right-leaning nodes
  if(is_red(h->right) && !is_red(h->left)) {
    changed = true;
    rotate_left(h);
  }

  // Fix two red nodes in a row
  if(is_red(h->left) && is_red(h->left->left)) {
    changed = true;
    rotate_right(h);
  }

  // Split any 4 nodes
  if(is_red(h->left) && is_red(h->right)) {
    changed = true;
    flip_color(*h);
  }

  if(!h->parent) h->set_color(Color::BLACK); // Reset root node to black

  return changed;
}

template<class T> auto RankedTree<T>::fixup_ancestors(NodeLink& mutated) -> void {
  NodeLink* iter = &mutated;
  int consecutive_unchanged_links = fixup(*iter) ? 0 : 1;
  while(iter != &m_root) {
    iter = &parent_link(*(*iter)->parent);
    if(fixup(*iter)) {
      consecutive_unchanged_links = 0;
    }
    else {
      // Properties that fixup is trying to repair depend on 3 consecutive levels of tree
      // If we didn't change this level or the previous two, we don't need to worry about other ancestors
      if(consecutive_unchanged_links >= 2)
        return;
      ++consecutive_unchanged_links;
    }
  }
}

// Sets h to new_node and then re-balances the tree
template<class T> auto RankedTree<T>::link_node(Node* parent, NodeLink& h, NodeLink&& new_node) -> void {
  assert(!h); // h should be empty
  assert(parent ? (&h == &parent->left || &h == &parent->right)
                : (&h == &m_root)); // h should be a link from the indicated parent (or the root link)
  assert(new_node); // We expect to be getting a non-null node to insert
  assert(!new_node->left && !new_node->right); // This function assumes new_node doesn't already have children which might unbalance tree more than could be fixed here
  assert(!new_node->parent); // We assume new_node doesn't already have some parent it needs to be detached from
  assert(new_node->rank() == 1); // Rank should be initialized to one
  assert(new_node->is_red()); // New nodes are red
  h = std::move(new_node);
  h->parent = parent;
  assert(h->parent != h.get()); // Check that we didn't try to make a node it's own parent
  // Update ranks all the way back to the root
  // TODO: It would probably be faster to combine this with fixup_ancestors
  for(Node* iter = h.get(); iter->parent; iter = iter->parent) {
    if(iter->is_left()) iter->parent->update_rank(1);
  }

  // We've attached a new red node so bubble up any fixes we need to make to avoid double reds or other constraints
  fixup_ancestors(h);
}

template<class T> auto RankedTree<T>::insert_after(const Iterator pos, NodeLink&& new_node) -> Iterator {
  const auto result = Iterator{new_node.get()};
  assert(pos); // Shouldn't try to insert after end
  assert(!empty()); // If pos is valid and this is empty, something went wrong (maybe pos is from a different tree?)

  // Inserting after end isn't intentionally supported, but seems safer to stuff value at end instead of dereferencing a null pointer
  if(!pos) return insert_before(end(), std::move(new_node));

  // Find node with empty link to attach to
  Node* iter = pos.node_ptr;
  while(iter->right) {
    assert(iter->left); // Left leaning
    iter = iter->left.get();
  }

  // Attaching on right temporarily violates left leaning property, but link_node will fix that
  link_node(iter, iter->right, std::move(new_node));
  assert(pos.node_ptr->next() == result.node_ptr);
  assert(result.node_ptr->prev() == pos.node_ptr);
  return result;
}

template<class T> auto RankedTree<T>::insert_before(const Iterator pos, NodeLink&& new_node) -> Iterator {
  if(empty()) { // Special case handling for empty tree
    assert(!pos); // It shouldn't be possible to have a valid iterator into an empty tree
    m_root = std::move(new_node);
    m_root->set_color(Color::BLACK);
    return Iterator{m_root.get()};
  }
  const auto result = Iterator{new_node.get()};

  Node* iter = pos.node_ptr;

  if(!iter) { // Special case handling for end of tree
    iter = &(m_root->max_child());
    assert(iter); // Should have caught empty tree above
    // Use insert_after to find correct spot
    return insert_after(Iterator{iter}, std::move(new_node));
  }

  if(!iter->left) {
    link_node(iter, iter->left, std::move(new_node));
  }
  else {
    iter = &(iter->left->max_child());
    // Attaching on right temporarily violates left leaning property, but link_node will fix that
    link_node(iter, iter->right, std::move(new_node));
  }
  assert(result.node_ptr->next() == pos.node_ptr);
  return result;
}

template<class T> auto RankedTree<T>::find_last() -> Iterator {
  if(!m_root) return Iterator{nullptr};
  else return Iterator{&(m_root->max_child())};
}

template<class T> auto RankedTree<T>::find_last() const -> ConstIterator {
  if(!m_root) return ConstIterator{nullptr};
  else return ConstIterator{&(m_root->max_child())};
}

template<class T> auto RankedTree<T>::empty() const -> bool { return m_root == nullptr; }
template<class T> auto RankedTree<T>::begin() const -> ConstIterator { return ConstIterator{m_root ? &m_root->min_child() : nullptr}; }
template<class T> auto RankedTree<T>::begin() -> Iterator { return Iterator{m_root ? &m_root->min_child() : nullptr}; }
template<class T> auto RankedTree<T>::end() const -> ConstIterator { return Iterator{nullptr}; }
template<class T> auto RankedTree<T>::end() -> Iterator { return Iterator{nullptr}; }

template<class T> auto RankedTree<T>::erase(ConstIterator pos) -> Iterator {
  assert(pos);
  const auto result = Iterator{pos.node().next()};
  extract_node(pos.node_ptr);
  return result;
}

template<class T> auto RankedTree<T>::find_index(const size_t i) -> Iterator {
  size_t m = i + 1;
  Node* iter = m_root.get();
  while(iter) {
    if(m < iter->rank()) {
      iter = iter->left.get();
    }
    else if(m > iter->rank()) {
      m -= iter->rank();
      iter = iter->right.get();
    }
    else {
      break;
    }
  }
  return Iterator{iter};
}

// Find iterator such that predicate returns true for all elements from [begin(),iterator)
// If p(*iterator) is true for all elements, this will be end
// Otherwise p(*iterator) will be false and iterator will point to beginning of list or previous element will be true
template<class T> template<class UnaryPredicate> auto RankedTree<T>::find_first_false(const UnaryPredicate& p) const -> ConstIterator {
  Node* iter = m_root.get();
  Node* result = nullptr;
  while(iter) {
    if(p(iter->cdata())) {
      iter = iter->right.get(); // Step forward
    }
    else {
      result = iter; // Save the candidate
      iter = iter->left.get(); // Step backwards
    }
  }
  return ConstIterator{result};
}

template<class T> template<class SignedPredicate> auto RankedTree<T>::equal_range(const SignedPredicate& p) const -> Range<ConstIterator> {
  Node* iter = m_root.get();
  if(!iter) return {end(), end()}; // Empty range if no root

  const auto scan_left = [&p](Node* iter) {
    return iter;
  };
  const auto scan_right = [&p](Node* iter) {
    while(iter->right && p(iter->right->cdata()) == 0)
      iter = iter->right;
    return iter;
  };

  for(;;) {
    const int comp = p(iter->cdata());

    if(comp > 0) {
      // Target range is strictly to the left of iter
      if(!iter->left) {
        // Nothing to the left so result is empty range at start of list
        assert(iter->prev() == nullptr);
        return {Iterator{iter},Iterator{iter}};
      }
      iter = iter->left.get();
    }
    else if(comp < 0) {
      // Target range is strictly to the right of iter
      if(!iter->right) {
        // Nothing to the right so result is empty range at end of list
        assert(iter->next() == nullptr);
        return {end(), end()};
      }
      iter = iter->right.get();
    }
    else {
      Node* lo_iter = iter;
      Node* hi_iter = iter;
      while(lo_iter->left && p(lo_iter->left->cdata()) == 0)
        lo_iter = lo_iter->left.get();
      while(hi_iter->right && p(hi_iter->right->cdata()) == 0)
        hi_iter = hi_iter->right.get();
      assert(hi_iter->next() == hi_iter->right.get()); // Search started at root so if we stopped at a leaf it must be the max leaf
      hi_iter = hi_iter->right.get(); // Bump hi side up by one so to make an open range
      return {Iterator{lo_iter}, Iterator{hi_iter}};
    }
  }
}



template<class T> auto RankedTree<T>::force_red(Node& node) -> void {
  if(node.is_red()) {
    return;
  }
  if(node.is_root()) {
    // Can safely swap root color
    node.set_color(Color::RED);
    return;
  }
  assert(node.is_black()); // Not red so must be black
  assert(node.parent->left && node.parent->right); // Black nodes should have a sibling
  if(node.parent->left->is_black() && node.parent->right->is_black()) {
    force_red(*node.parent);
    flip_color(*node.parent);
    return;
  }
  // Sibling must be red (since this node is black and nodes aren't both black)
  // From left leaning property this node must also be on the right
  assert(node.is_right());
  assert(node.parent->left->is_red());
  rotate_right(parent_link(*node.parent));
  // We just broke left leaning property on parent, but now we fix it
  assert(node.parent->is_red());
  assert(node.parent->left && node.parent->left->is_black());
  assert(node.parent->right.get() == &node);
  assert(node.is_black()); // Still
  flip_color(*node.parent); // Now we can make this node red
  assert(node.is_red());
}

template<class T> auto RankedTree<T>::extract_node(Node* node) -> void {
  assert(node);
  assert(!node->right || (node->right && node->left)); // Check that node is left-leaning since we're about to use that fact

  if(node->left) {
    // Node has children. Need to shuffle things around so that we are erasing a node without children
    Node& alternate = node->left->min_child(); // Find some node below h without children
    assert(!alternate.left && !alternate.right); // Since tree is left-leaning, min_child should have no children
    // Now we swap position of alternate with position of h
    swap_positions(alternate, *node);
  }

  // Step above ensures we are only trying to erase a node without children
  assert(!node->left && !node->right);

  force_red(*node);

  // Update rank for all ancestors
  for(Node* iter = node; iter->parent; iter = iter->parent) {
    if(iter->is_left()) iter->parent->update_rank(-1);
  }

  if(node->is_root()) {
    assert(m_root.get() == node);
    // Delete the node by resetting the root
    m_root.reset();
    // No cleanup needed since tree is now empty
  }
  else {
    // Grab link to the node that we're about to destroy
    NodeLink& link_to_node = parent_link(*node);
    // Grab link above that which we can use for cleanup
    NodeLink& link_to_parent = parent_link(*node->parent);
    // Delete the node by resetting the link to it
    link_to_node.reset();

    // Cleanup as needed back to the root
    fixup_ancestors(link_to_parent);
  }
}

} // geode namespace