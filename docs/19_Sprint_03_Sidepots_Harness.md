# Sprint 03 — Side Pots, Harness, Regression

## Objective
Close the reproducibility loop and improve settlement correctness for short-stack all-in situations while adding a repeatable local benchmark.

## Delivered
- uncalled all-in refund before showdown
- pot segmentation snapshot for auditability
- baseline agent vs baseline agent harness
- smoke benchmark service and API endpoint
- golden hands regression fixture set

## Important scope note
In heads-up no-limit, classic multi-side-pot branching is limited compared with 3+ player games. This sprint therefore targets the realistic heads-up requirement: **correct settlement and refund handling when stack depths differ**.
