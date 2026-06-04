# Observation Features

The policy observes a compact numeric state vector rather than images. These
features describe the car's current motion, alignment to the track, upcoming
track geometry, and progress around the lap.

## Observation V1

Observation V1 contains 12 `float32` features. Most values are normalised to
`[-1, 1]`; `progress_ratio` is normalised to `[0, 1]`.

| Index | Feature | Meaning |
| ---: | --- | --- |
| 0 | `forward_speed_norm` | Vehicle-frame forward speed, including reversing when negative. |
| 1 | `lateral_speed_norm` | Vehicle-frame sideways speed, useful for detecting sliding. |
| 2 | `vertical_speed_norm` | Up/down speed from jumps, crests, or suspension movement. |
| 3 | `heading_error_norm` | Difference between car heading and current track direction. |
| 4 | `lateral_error_norm` | Signed distance from the centreline; positive is left of the centreline. |
| 5 | `lookahead_heading_error_20m_norm` | Heading error to the centreline point 20 m ahead. |
| 6 | `lookahead_heading_error_40m_norm` | Heading error to the centreline point 40 m ahead. |
| 7 | `lookahead_heading_error_80m_norm` | Heading error to the centreline point 80 m ahead. |
| 8 | `curvature_20m_norm` | Track curvature 20 m ahead. |
| 9 | `curvature_40m_norm` | Track curvature 40 m ahead. |
| 10 | `curvature_80m_norm` | Track curvature 80 m ahead. |
| 11 | `progress_ratio` | Fractional raw lap progress along the closed centreline. |

## Observation V2

Observation V2 appends the previous action to the V1 vector, giving 14 features
total:

| Index | Feature | Meaning |
| ---: | --- | --- |
| 12 | `prev_steer_norm` | Steering command from the previous policy step. |
| 13 | `prev_throttle_brake_norm` | Previous combined throttle/brake command. Positive means throttle; negative means brake. |

The previous-action features give the policy limited control history. They were
added to help reduce steering oscillation in continuous-control runs without
requiring a recurrent policy.

## Normalisation Constants

| Quantity | Scale |
| --- | ---: |
| Forward speed | `60.0` m/s |
| Lateral speed | `20.0` m/s |
| Vertical speed | `10.0` m/s |
| Lateral error | `10.0` m |
| Heading error | `pi` radians |
| Curvature | `0.05` 1/m |

Values outside these ranges are clipped so the observation space remains stable
for PPO.
