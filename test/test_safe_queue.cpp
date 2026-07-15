#include <cassert>
#include <optional>

#include "common/SafeQueue.hpp"

int main() {
    SafeQueue<int> queue{2};

    assert(!queue.push_latest(1).has_value());
    assert(!queue.push_latest(2).has_value());

    const std::optional<int> dropped = queue.push_latest(3);
    assert(dropped.has_value());
    assert(*dropped == 1);
    assert(queue.size() == 2);

    int value = 0;
    assert(queue.pop(value));
    assert(value == 2);
    assert(queue.pop(value));
    assert(value == 3);

    queue.close();
    assert(!queue.pop(value));
    const std::optional<int> rejected = queue.push_latest(4);
    assert(rejected.has_value());
    assert(*rejected == 4);
    assert(queue.empty());

    queue.reopen();
    assert(!queue.push_latest(5).has_value());
    assert(queue.pop(value));
    assert(value == 5);
    return 0;
}
