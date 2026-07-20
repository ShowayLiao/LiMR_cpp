#include "dashboard_task_state.h"

#include <cassert>

int main() {
    assert(!dashboard_task_is_busy(DashboardTaskState::Idle));
    assert(dashboard_task_is_busy(DashboardTaskState::Running));
    assert(!dashboard_task_is_busy(DashboardTaskState::Succeeded));
    assert(!dashboard_task_is_busy(DashboardTaskState::Failed));
    return 0;
}
