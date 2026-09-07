# Claude worker failure recovery

An assigned worker may return its initial result and later fail while reading a follow-up. Before this correction the launcher returned the child exit code without an error marker once *any* assignment result had been seen. The buffered daemon stdin reader could then abort Python at shutdown, and the structured status parser could keep the earlier completion or classify the echoed input as processing. The caller waited without a failure notice.

The launcher now uses a cancellable file-descriptor relay and joins it before interpreter exit. Nonzero child exit after an earlier result emits a follow-up error marker. Structured status gives a subsequent error precedence over an earlier completion and recognizes the observed legacy parser/shutdown diagnostic lines. JSON model content containing those phrases is not a terminal error.

The server commits a separate idempotent SYSTEM inbox notice, keyed by completion identity, when an assigned worker reaches ERROR, including a retained worker whose original completion is already acknowledged. Successful reports are never rewritten. Failed enqueue is retried through the existing server scheduler; startup reconciliation scans retained assignments so failure notification can recover after restart. No model is polled and no task, collector, or replacement is automatically retried.

Tests include a real launcher subprocess whose child exits after a result while launcher stdin remains open, status precedence, initial and follow-up failure notices, duplicate delivery, database retry, restart recovery, and existing inbox/provider regressions. They use temporary CAO state and no model/API acquisition.

## Operational deployment

Use the reviewed maintenance-branch revision, preserving existing unrelated worktrees and the installed dependency versions. Retain the prior four Python module files before replacing the installed package or modules. No CAO database schema migration is required.

Check the service and tmux-server cgroups before restarting. If tmux-server belongs to cao-server.service, the default KillMode=control-group would kill the attached sessions. The included deploy/systemd/cao-server-preserve-tmux.conf is a service drop-in for this deployment shape: KillMode=process allows persistent tmux sessions to survive the API restart. Install the drop-in, daemon-reload, verify the effective KillMode, then restart only the CAO API. Do not restart before that check passes. Shutdown/retirement of terminal sessions remains an explicit CAO operation.

After restart verify API health, the original Brain terminal/session, ERROR status for the failed worker, one SYSTEM failure notice delivered to the original caller, and unchanged Atlas DB/ledger fingerprints. Do not type into the failed worker's bare shell. Preserve its report and create a replacement only under the original authorized task scope.

Rollback restores the exact backed-up Python modules and restarts the API with the session-preserving drop-in still active. Removing the drop-in is a separate operation requiring terminal-lifetime review.

## Limits

The observed user JSON echoed in the terminal log was valid. The exact cause of Claude Code's native SyntaxError has not been established; this correction must not be presented as a proven fix inside the Claude binary. It fixes the subsequent silent-wait and Python-shutdown failures. Arbitrary SIGKILL of the launcher or total host failure cannot emit its marker; retained evidence and service reconciliation remain necessary. A new error notification is operational evidence, not permission to resume financial acquisition.
