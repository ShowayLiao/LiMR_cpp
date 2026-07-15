# Native file picker design

## Goal

Replace the need to manually type input-media and model paths in the dashboard with Windows native file-selection dialogs, while retaining editable path fields.

## UI

The existing `Video` and `Model Path` text fields remain unchanged. Each gets a neighbouring `Browse...` button:

- `Video`: opens a file dialog limited to supported images and videos, with an all-files fallback.
- `Model Path`: opens a file dialog limited to ONNX and TensorRT engine files, with an all-files fallback.

Selecting a file copies its absolute path into the associated ImGui buffer. Cancelling leaves its current value unchanged.

## Implementation

Use the Windows common `GetOpenFileNameA` dialog. Keep the dialog invocation in a small, independently testable helper that accepts an initial path and a filter, and only modifies the supplied destination when the dialog reports success. The dashboard supplies the two filters and copies a selected path into its existing fixed-size buffers.

## Verification

Add a unit test for the path-copy helper to prove successful selections are copied safely and cancellation preserves the prior path. Build the test target and the application target to confirm the Win32 dependency and ImGui UI changes compile.
