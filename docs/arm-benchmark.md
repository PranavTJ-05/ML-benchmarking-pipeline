# ARM benchmark procedure

Run manually on a physical Raspberry Pi when available. Otherwise use TFLite's
benchmark tooling on an Android device over `adb`. Record device model, exact
SoC, OS, runtime, threads, config, and artifact checksum in the JSON result.

Do not use QEMU or other ARM emulation for reported device numbers. CI validates
committed ARM JSON against the result schema but does not reproduce ARM latency
on shared x86 runners.
