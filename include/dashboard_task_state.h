#pragma once

enum class DashboardTaskState {
    Idle,
    Running,
    Succeeded,
    Failed,
};

constexpr bool dashboard_task_is_busy(DashboardTaskState state) {
    return state == DashboardTaskState::Running;
}
