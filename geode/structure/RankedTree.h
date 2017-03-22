#pragma once
#include <geode/config.h>
#include <geode/math/max.h>
#include <geode/utility/debug.h>
#include <geode/utility/range.h>
#include <memory>
#include <iostream>

namespace geode {

// Stores a collection of elements in a fixed order with support for the following operation:
//   Insertion, and removal of elements from anywhere in log time
//   Reorder any two elements in constant time
//   Binary search over the entire collection (assuming it is sorted) in log time
//   Iterate over consecutive elements in amortized constant time
//   Find an element at a specific index in log time
// This class is intended to store a sorted collection updating it as elements are added and removed or as sort order changes
// Unlike std::set/map/multiset/multimap this class does not maintain ordered keys for entries, allowing for a comparison function that might change
// (Though the assumption is that any changes to comparison function will only alter a small number of nodes)
// Caller is responsible for updating nodes when their order changes for binary search to work
template<class T> class RankedTree {
 private:
  class Node;

  class IteratorBase {
   protected:
    friend class RankedTree;
    // Traversal order should be left, self, right
    Node* node_ptr = nullptr;
    Node& node() const { assert(node_ptr); return *node_ptr; }
    explicit IteratorBase(Node* n) : node_ptr(n) { }
   public:
    IteratorBase() = default;
    explicit operator bool() const { return node_ptr != nullptr; }
    inline friend bool operator==(const IteratorBase& lhs, const IteratorBase& rhs) { return lhs.node_ptr == rhs.node_ptr; }
    inline friend bool operator!=(const IteratorBase& lhs, const IteratorBase& rhs) { return lhs.node_ptr != rhs.node_ptr; }
  };
 public:
  class ConstIterator : public IteratorBase {
   protected:
    friend class RankedTree;
    friend class Iterator;
    explicit ConstIterator(Node* n) : IteratorBase(n) { }
    using IteratorBase::node_ptr;
   public:
    ConstIterator& operator--() { assert(node_ptr); node_ptr = node_ptr->prev(); return *this; }
    ConstIterator& operator++() { assert(node_ptr); node_ptr = node_ptr->next(); return *this; }
    ConstIterator prev() const { assert(node_ptr); return ConstIterator{node_ptr->prev()}; }
    ConstIterator next() const { assert(node_ptr); return ConstIterator{node_ptr->next()}; }
    const T& operator*() const { assert(node_ptr); return node_ptr->data; }
  };
  class Iterator : public IteratorBase {
   protected:
    friend class RankedTree;
    explicit Iterator(Node* n) : IteratorBase(n) { }
    using IteratorBase::node_ptr;
   public:
    // Allow implicit conversion to ConstIterator
    operator ConstIterator() const { return ConstIterator{node_ptr}; }
    Iterator& operator--() { assert(node_ptr); node_ptr = node_ptr->prev(); return *this; }
    Iterator& operator++() { assert(node_ptr); node_ptr = node_ptr->next(); return *this; }
    Iterator prev() const { assert(node_ptr); return Iterator{node_ptr->prev()}; }
    Iterator next() const { assert(node_ptr); return Iterator{node_ptr->next()}; }
    T& operator*() const { assert(node_ptr); return node_ptr->data; }
  };


  bool empty() const;
  Iterator begin();
  ConstIterator begin() const;
  Iterator end();
  ConstIterator end() const;

  Iterator find_last();
  ConstIterator find_last() const;

  // Inserts a new element directly before pos, returns iterator to newly inserted element. O(log(size))
  // No iterators or references are invalidated
  template<class... Args> Iterator emplace(const ConstIterator pos, Args&&... args)
  { return insert_before(pos, NodeLink{new Node{std::forward<Args>(args)...}}); }
  template<class... Args> Iterator emplace_after(const ConstIterator pos, Args&&... args)
  { return insert_after(pos, NodeLink{new Node{std::forward<Args>(args)...}}); }
  template<class... Args> Iterator emplace_back(Args&&... args)
  { return insert_before(end(), NodeLink{new Node{std::forward<Args>(args)...}}); }

  // Removes element at pos. Returns iterator to next element after removed iterator
  // Only iterators pointing to erased element should be invalidated
  Iterator erase(const ConstIterator pos);

  // Swaps position within list of two elements
  // No iterators are invalidated. Addresses do not change
  void swap_positions(const ConstIterator pos0, const ConstIterator pos1) { assert(pos0 && pos1); swap_positions(pos0.node(), pos1.node()); }

  // Locate element at index i. O(log(size))
  Iterator find_index(size_t i);
  ConstIterator find_index(size_t i) const;

  // The following functions assume element are partially ordered with respect to searched element or predicate
  // i.e. Order should be as if std::partition has been called with predicate

  // Finds first element for which p returns false (p)
  // Returns end if no such element exists
  template<class UnaryPredicate> ConstIterator find_first_false(const UnaryPredicate& p) const;
  template<class UnaryPredicate> Iterator find_first_false(const UnaryPredicate& p)
  { return Iterator{static_cast<const RankedTree*>(this)->find_first_false(p).node_ptr}; }
  // Assuming p returns <0 for elements before range, ==0 for elements inside range, and >0 for elements after range
  // Finds [start, end) containing all elements that return ==0
  template<class SignedPredicate> Range<ConstIterator> equal_range(const SignedPredicate& p) const;


  // Inserts a new element before first element for which is_before(element) is false
  template<class UnaryPredicate, class... Args> Iterator emplace_in_order(const UnaryPredicate& is_before, Args&&... args)
  { return insert_before(find_first_false(is_before), NodeLink{new Node{std::forward<Args>(args)...}}); }

  // Dump internal data for debugging
  void debug_print(std::ostream& os) const { debug_print(os, '_', 0, m_root.get()); }

  // Verifies expected properties for every node
  void test_global_invarients() const;
 private:
  // Implementation is based on a left leaning red black tree with a 'rank' added to each node that can be converted to an index during iteration
  // 'rank' tracks total number of nodes in left subtree which is updated as part of any re-balancing operations
  // Ownership of nodes is managed via std::unique_ptr in parent node or at root of tree

  // Enums for tracking properties of 
  enum class Color : bool { RED, BLACK };
  enum class Side : bool { LEFT, RIGHT };
  inline friend Color operator!(Color c) { return static_cast<Color>(!static_cast<bool>(c)); }
  inline friend Side operator^(Side s, bool b) { return static_cast<Side>(static_cast<bool>(s) ^ b); }
  using NodeLink = std::unique_ptr<Node>;
  static bool is_red(Node* n) { return n && n->is_red(); }
  static bool is_black(Node* n) { return !n || n->is_black(); }
  static bool is_red(const NodeLink& n) { return is_red(n.get()); }
  static bool is_black(const NodeLink& n) { return is_black(n.get()); }
  class Node {
   public:
    NodeLink left; // Left nodes come before this
    NodeLink right; // Right nodes come after this
    Node* parent = nullptr;
    static constexpr size_t combine(size_t rank, Color c) { return (rank<<1)|static_cast<bool>(c); }
    size_t m_rank_and_color = combine(1, Color::RED);
    T data;
    const T& cdata() const { return data; } // Shorthand for casting data to const
    // Constructor forwards all arguments to T
    template<class... Args> explicit Node(Args&&... args)
     : data(std::forward<Args>(args)...)
    { }
    ~Node() { assert(!parent || (parent->left.get() != this && parent->right.get() != this)); }

    size_t rank() const { return m_rank_and_color>>1; }
    void update_rank(ssize_t delta) { assert(rank() + delta > 0); m_rank_and_color += (delta<<1); }
    Color color() const { return static_cast<Color>(m_rank_and_color & 1); }
    void set_color(const Color new_color) { m_rank_and_color = combine(rank(), new_color); }
    void set_rank(const size_t new_rank) { m_rank_and_color = combine(new_rank, color()); }

    bool is_red() const { return color() == Color::RED; }
    bool is_black() const { return color() == Color::BLACK; }
    static bool is_red(Node* n) { return RankedTree::is_red(n); }
    static bool is_black(Node* n) { return RankedTree::is_black(n); }

    bool is_root() const { return !parent; }
    bool is_left() const { assert(parent && ((parent->left.get() == this) != (parent->right.get() == this))); return (parent->left.get() == this); }
    bool is_right() const { assert(parent && ((parent->left.get() == this) != (parent->right.get() == this))); return (parent->right.get() == this); }

    Side side() const;
    NodeLink link(const Side s) { return (s == Side::LEFT) ? left : right; }

    Node& min_child() { return left ? left->min_child() : *this; }
    Node& max_child() { return right ? right->max_child() : *this; }

    Node* next(); // Find first node after this or null if this node is the max
    Node* prev(); // Previous node before this or null if this node is the min

    // Validates expected invariants for an individual node and connections to neighbors
    void test_local_invarients() const;
  };

  NodeLink m_root;

  // Returns the link that owns node. This will be either m_root, node.parent->left, or node.parent->right
  NodeLink& parent_link(Node& node);

  // Swaps all auxiliary data and references to/from n0/n1 but leaves Node::data unchanged
  // This could almost be replaced with swap(n0.data, n1.data), but that invalidates iterators and requires T is swappable
  void swap_positions(Node& n0, Node& n1);

  // Swaps h and h->right adjusting other members accordingly
  static void rotate_left(NodeLink& h);
  // Swaps h and h->left adjusting other members accordingly
  static void rotate_right(NodeLink& h);
  // Flips color of h and children of h
  static void flip_color(Node& node);
  // Rotates h as needed to maintain invariants after inserting or deleting a node
  static bool fixup(NodeLink& h);
  // Makes node red without unbalancing tree. Might leave ancestors right-leaning
  void force_red(Node& node);
  // Walks from h to root of tree handling needed changes after inserting or deleting a node
  void fixup_ancestors(NodeLink& h);

  // Moves new_node into h then walks up tree re-balancing nodes as needed
  // Requires h is a currently empty leaf of the tree
  void link_node(Node* parent, NodeLink& h, NodeLink&& new_node);

  // Helper functions that find suitable parent node and then call link_node
  Iterator insert_before(const Iterator pos, NodeLink&& new_element);
  Iterator insert_after(const Iterator pos, NodeLink&& new_element);

  // Removes node from the tree, reattaching children in the tree in the correct order and re-balancing as needed
  // Node must be valid when calling the function but will be deleted by the time this function returns
  void extract_node(Node* node);

  struct SubtreeStats {
    size_t black_height; // Total number of black links in this subtree (should be the same for any path to a leaf)
    size_t total_size; // Total count of nodes in this subtree
    size_t max_height; // Longest path to a leaf from root of this subtree
  };
  // 
  SubtreeStats test_subtree_invarients(const Node* node) const; 

  void debug_print(std::ostream& os, const char prefix, const int indent, const Node * node) const;
};

// This printing operator includes a bunch of debug information and might not be ideal for general use
template<class T> std::ostream& operator<<(std::ostream& os, const RankedTree<T>& tree)
{ tree.debug_print(os); return os; }

} // geode namespace

#include "RankedTree_p.h"
