Raw surgical videos are not publicly released because of hospital ethics and patient privacy restrictions. Training and evaluation are performed on the hospital intranet.

Each .pt clip is expected to contain frames tensor [3,16,480,640]; motion [16,6]; detection_conf [16]; label_cls int (0=novice, 1=intermediate, 2=expert); label_reg [7] (6 GRS dimensions plus total, normalised to 0-1); and centre str ("A"|"B"|"C").

See [../RESULTS.md](../RESULTS.md) for reported results and the statistical audit trail.
