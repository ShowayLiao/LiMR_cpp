#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <cstddef>

template<typename T>
class SafeQueue {
public:
    explicit SafeQueue(size_t capacity = 0) : capacity_(capacity) {}
    ~SafeQueue() = default;

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    // For real-time pipelines, retain the newest frames instead of accumulating latency.
    // Returns the item that was not retained: the oldest item when full, or value when closed.
    std::optional<T> push_latest(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return std::optional<T>(std::move(value));

        std::optional<T> discarded;
        if (capacity_ != 0 && queue_.size() >= capacity_) {
            discarded.emplace(std::move(queue_.front()));
            queue_.pop();
        }
        queue_.push(std::move(value));
        cond_.notify_one();
        return discarded;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || closed_; });
        
        if (queue_.empty()) {
            return false;
        }
        
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cond_.notify_all();
    }

    void shutdown() { close(); }

    void reopen() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = false;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = false;
        std::queue<T> empty;
        queue_.swap(empty);
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    std::queue<T> queue_;
    size_t capacity_ = 0;
    bool closed_ = false;
};
