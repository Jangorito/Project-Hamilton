angular.module('beamng.apps')
.directive('beamngRlTrainingHud', [function () {
  return {
    template:
    `<div class="beamng-rl-hud bngApp">
      <style>
        .beamng-rl-hud {
          width: 100%;
          height: 100%;
          overflow: hidden;
          padding: 12px;
          color: #eef5ff;
          background: rgba(6, 10, 16, 0.78);
          border: 1px solid rgba(140, 180, 230, 0.32);
          border-radius: 6px;
          box-shadow: 0 10px 26px rgba(0, 0, 0, 0.42);
          font-family: "Segoe UI", Arial, sans-serif;
        }
        .beamng-rl-hud * { box-sizing: border-box; }
        .beamng-rl-hud__top {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 8px;
          margin-bottom: 8px;
        }
        .beamng-rl-hud__title {
          font-size: 13px;
          font-weight: 700;
          line-height: 1.1;
        }
        .beamng-rl-hud__sub {
          margin-top: 2px;
          color: #a9b7c8;
          font-size: 10px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          max-width: 280px;
        }
        .beamng-rl-hud__status {
          padding: 3px 7px;
          border-radius: 999px;
          font-size: 10px;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: #8ff0a4;
          background: rgba(30, 120, 60, 0.35);
          border: 1px solid rgba(90, 220, 120, 0.45);
        }
        .beamng-rl-hud__status.is-idle {
          color: #c2cad5;
          background: rgba(80, 90, 105, 0.35);
          border-color: rgba(170, 185, 205, 0.3);
        }
        .beamng-rl-hud__bar {
          height: 7px;
          width: 100%;
          overflow: hidden;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.14);
          margin: 4px 0 8px;
        }
        .beamng-rl-hud__bar-fill {
          height: 100%;
          width: 0%;
          border-radius: 999px;
          background: linear-gradient(90deg, #38bdf8, #7dd3fc);
          transition: width 0.2s ease;
        }
        .beamng-rl-hud__bar-fill.is-lap {
          background: linear-gradient(90deg, #facc15, #fb923c);
        }
        .beamng-rl-hud__grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 6px;
          margin-bottom: 8px;
        }
        .beamng-rl-hud__metric {
          min-width: 0;
          padding: 6px 7px;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 5px;
        }
        .beamng-rl-hud__label {
          color: #9aa9ba;
          font-size: 9px;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          white-space: nowrap;
        }
        .beamng-rl-hud__value {
          margin-top: 2px;
          font-size: 15px;
          line-height: 1.05;
          font-weight: 800;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .beamng-rl-hud__value.small {
          font-size: 12px;
          line-height: 1.2;
        }
        .beamng-rl-hud__section {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          margin-top: 8px;
        }
        .beamng-rl-hud__panel {
          min-width: 0;
          padding: 7px;
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 5px;
        }
        .beamng-rl-hud__row {
          display: flex;
          justify-content: space-between;
          gap: 8px;
          margin-top: 4px;
          color: #dbe7f3;
          font-size: 10px;
        }
        .beamng-rl-hud__row span:first-child {
          color: #9aa9ba;
        }
        .beamng-rl-hud__warn {
          color: #facc15;
        }
        .beamng-rl-hud__bad {
          color: #fb7185;
        }
        .beamng-rl-hud__good {
          color: #8ff0a4;
        }
        .beamng-rl-hud__empty {
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #a9b7c8;
          font-size: 12px;
          text-align: center;
        }
      </style>

      <div ng-if="!data || !data.has_data" class="beamng-rl-hud__empty">
        Waiting for Project Hamilton telemetry...
      </div>

      <div ng-if="data && data.has_data">
        <div class="beamng-rl-hud__top">
          <div>
            <div class="beamng-rl-hud__title">Project Hamilton Training</div>
            <div class="beamng-rl-hud__sub">{{ data.run_name || "run" }} | {{ data.vehicle_label || data.vehicle_model }} | {{ data.reward_config }}</div>
          </div>
          <div class="beamng-rl-hud__status" ng-class="{'is-idle': !data.active}">
            {{ data.status || "training" }}
          </div>
        </div>

        <div class="beamng-rl-hud__row">
          <span>Training</span>
          <span>{{ fmtInt(data.current_step) }} / {{ fmtInt(data.total_steps) }} steps - {{ fmtPct(data.step_percent) }} - {{ fmtEta(data.eta_seconds) }}</span>
        </div>
        <div class="beamng-rl-hud__bar">
          <div class="beamng-rl-hud__bar-fill" ng-style="{width: clampPct(data.step_percent)}"></div>
        </div>

        <div class="beamng-rl-hud__grid">
          <div class="beamng-rl-hud__metric">
            <div class="beamng-rl-hud__label">Episode</div>
            <div class="beamng-rl-hud__value">{{ fmtInt(data.episode_number) }}</div>
          </div>
          <div class="beamng-rl-hud__metric">
            <div class="beamng-rl-hud__label">Ep Step</div>
            <div class="beamng-rl-hud__value">{{ fmtInt(data.episode_step) }}</div>
          </div>
          <div class="beamng-rl-hud__metric">
            <div class="beamng-rl-hud__label">Speed km/h</div>
            <div class="beamng-rl-hud__value">{{ fmtNum(data.speed_kph, 0) }}</div>
          </div>
          <div class="beamng-rl-hud__metric">
            <div class="beamng-rl-hud__label">SPS</div>
            <div class="beamng-rl-hud__value">{{ fmtNum(data.steps_per_second, 1) }}</div>
          </div>
        </div>

        <div class="beamng-rl-hud__row">
          <span>Lap</span>
          <span>{{ fmtNum(data.episode_progress_m, 1) }} m - {{ fmtPct(data.lap_progress_percent) }} - best {{ fmtNum(data.best_episode_progress_m, 1) }} m</span>
        </div>
        <div class="beamng-rl-hud__bar">
          <div class="beamng-rl-hud__bar-fill is-lap" ng-style="{width: clampPct(data.lap_progress_percent)}"></div>
        </div>

        <div class="beamng-rl-hud__section">
          <div class="beamng-rl-hud__panel">
            <div class="beamng-rl-hud__label">Car State</div>
            <div class="beamng-rl-hud__row"><span>Lateral</span><span ng-class="riskClass(data.lateral_error_m, 4, 8)">{{ fmtNum(data.lateral_error_m, 2) }} m</span></div>
            <div class="beamng-rl-hud__row"><span>Heading</span><span ng-class="riskClass(abs(data.heading_error_deg), 10, 25)">{{ fmtNum(data.heading_error_deg, 1) }} deg</span></div>
            <div class="beamng-rl-hud__row"><span>Damage</span><span ng-class="riskClass(data.vehicle_damage, 100, 400)">{{ fmtNum(data.vehicle_damage, 0) }}</span></div>
            <div class="beamng-rl-hud__row"><span>Stuck</span><span>{{ fmtInt(data.stuck_steps) }}</span></div>
          </div>

          <div class="beamng-rl-hud__panel">
            <div class="beamng-rl-hud__label">Terminations</div>
            <div class="beamng-rl-hud__row"><span>Total</span><span>{{ fmtInt(data.terminations_total) }}</span></div>
            <div class="beamng-rl-hud__row"><span>Latest</span><span ng-class="data.latest_termination === 'none' ? 'beamng-rl-hud__good' : 'beamng-rl-hud__bad'">{{ data.latest_termination || "none" }}</span></div>
            <div class="beamng-rl-hud__row"><span>Off/Dam/Stuck</span><span>{{ fmtInt(data.term_counts.off_track) }} / {{ fmtInt(data.term_counts.damage) }} / {{ fmtInt(data.term_counts.stuck) }}</span></div>
            <div class="beamng-rl-hud__row"><span>Wall/Max</span><span>{{ fmtInt(data.term_counts.wall_bash) }} / {{ fmtInt(data.term_counts.max_episode_steps) }}</span></div>
          </div>
        </div>

        <div class="beamng-rl-hud__section">
          <div class="beamng-rl-hud__panel">
            <div class="beamng-rl-hud__label">Reward</div>
            <div class="beamng-rl-hud__row"><span>Last / Avg</span><span>{{ fmtNum(data.reward, 2) }} / {{ fmtNum(data.reward_mean, 2) }}</span></div>
            <div class="beamng-rl-hud__row"><span>Progress</span><span>{{ fmtNum(data.progress_reward, 2) }}</span></div>
            <div class="beamng-rl-hud__row"><span>Speed</span><span>{{ fmtNum(data.speed_reward, 2) }}</span></div>
          </div>

          <div class="beamng-rl-hud__panel">
            <div class="beamng-rl-hud__label">Control / Flags</div>
            <div class="beamng-rl-hud__row"><span>Steer</span><span>{{ fmtNum(data.steering, 2) }}</span></div>
            <div class="beamng-rl-hud__row"><span>Thr / Brk</span><span>{{ fmtNum(data.throttle, 2) }} / {{ fmtNum(data.brake, 2) }}</span></div>
            <div class="beamng-rl-hud__row"><span>Jumps</span><span ng-class="data.progress_jumps > 0 ? 'beamng-rl-hud__warn' : ''">{{ fmtInt(data.progress_jumps) }}</span></div>
          </div>
        </div>
      </div>
    </div>`,
    replace: true,
    restrict: 'EA',
    scope: true,
    controller: ['$scope', '$interval', function ($scope, $interval) {
      $scope.data = { has_data: false, active: false, term_counts: {} }
      var lastUpdate = 0

      function num(value, fallback) {
        var n = Number(value)
        return Number.isFinite(n) ? n : (fallback || 0)
      }

      $scope.fmtInt = function (value) {
        return Math.round(num(value, 0)).toLocaleString()
      }

      $scope.fmtNum = function (value, digits) {
        return num(value, 0).toFixed(digits)
      }

      $scope.fmtPct = function (value) {
        return num(value, 0).toFixed(1) + "%"
      }

      $scope.clampPct = function (value) {
        var pct = Math.max(0, Math.min(100, num(value, 0)))
        return pct.toFixed(1) + "%"
      }

      $scope.fmtEta = function (seconds) {
        var s = Math.max(0, Math.round(num(seconds, 0)))
        if (s <= 0) return "complete"
        var h = Math.floor(s / 3600)
        var m = Math.floor((s % 3600) / 60)
        if (h > 0) return h + "h " + m + "m left"
        return m + "m left"
      }

      $scope.riskClass = function (value, warn, bad) {
        var n = Math.abs(num(value, 0))
        if (n >= bad) return "beamng-rl-hud__bad"
        if (n >= warn) return "beamng-rl-hud__warn"
        return ""
      }

      $scope.abs = function (value) {
        return Math.abs(num(value, 0))
      }

      $scope.$on('BeamNGRLTrainingHUDData', function (event, data) {
        $scope.$evalAsync(function () {
          data = data || {}
          data = Object.assign({}, $scope.data || {}, data)
          data.has_data = true
          data.term_counts = data.term_counts || {}
          $scope.data = data
          lastUpdate = Date.now()
        })
      })

      var staleTimer = $interval(function () {
        if (!$scope.data.has_data || lastUpdate <= 0) return
        if (Date.now() - lastUpdate > 8000) {
          $scope.data.active = false
          $scope.data.status = "stale"
        }
      }, 1000)

      $scope.$on('$destroy', function () {
        $interval.cancel(staleTimer)
      })
    }]
  }
}]);
