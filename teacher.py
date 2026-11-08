import heapq

from sandbox.crane import action, forecast, me, paths, tile, visible
from sandbox.observation_types import (
    AxialPosition,
    SkirmishAction,
    SkirmishObservation,
)


class Season2CoreAgent:
    """
    Hybrid FSM + Behavior Tree + Utility AI controller for Season 2.

    High-level FSM:
        RENDEZVOUS -> FORM_UP -> COMBAT -> RECOVER

    Combat uses behavior-tree style priority rules.

    Special anti-cavalry logic:
        Enemy cavalry may act late in one round and early in the next,
        producing two attacks very close together. Therefore:

        * Archer treats ANY visible enemy cavalry as urgent.
        * Archer moves away while trying to put the footman between itself
          and the cavalry.
        * Footman screens/intercepts cavalry threatening the archer.
        * Friendly cavalry counter-pressures cavalry threatening the archer.

    Utility AI:
        * Scores enemy targets instead of relying on a single heuristic.
        * Gives the archer a strong Season 2 hill-seeking policy: establish
          nearby high ground before normal firing, and escape toward the nearest
          safe hill when an enemy reaches melee range.
        * Scores fallback archer escape/reposition paths using safety, support,
          terrain, center bias, and whether enemies have already acted.
        * Scores cavalry flank positions using access, flank angle, safety,
          support, and Season 2 terrain.
    """

    # ==========================================================
    # SHARED / FORMATION SETTINGS
    # ==========================================================

    RALLY_DISTANCE = 4

    # ==========================================================
    # ARCHER SETTINGS
    # ==========================================================

    ARCHER_RANGE = 6
    ARCHER_DAMAGE = 2
    ARCHER_IDEAL_DISTANCE = 5

    # Season 2 hill behavior. The archer strongly prefers to establish a
    # firing position on a nearby hill once combat begins. Strategic route
    # cost is used instead of straight-line distance so water cannot trap it.
    ARCHER_HILL_SEARCH_COST = 6.0
    ARCHER_EMERGENCY_HILL_SEARCH_COST = 8.0

    # Non-cavalry danger.
    ARCHER_MELEE_DANGER = 2

    # Archer vision is 6. Because cavalry can move 4 and may receive two
    # closely spaced activations across a round boundary, ANY visible
    # cavalry is treated as an urgent threat.
    ARCHER_CAVALRY_THREAT = 6

    # ==========================================================
    # FOOTMAN SETTINGS
    # ==========================================================

    FOOTMAN_DAMAGE = 3
    FOOTMAN_RANGE = 1

    # ==========================================================
    # CAVALRY SETTINGS
    # ==========================================================

    CAVALRY_DAMAGE = 3
    CAVALRY_RANGE = 1
    CAVALRY_RESET_DISTANCE = 4

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self, seed, observation) -> None:
        self.phase = "RENDEZVOUS"
        self.combat_state = None
        self.has_seen_enemy = False
        self.current_target_id = None

    # ==========================================================
    # MAIN DECISION FUNCTION
    # ==========================================================

    def act(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:

        unit_type = me.unit_type(observation)
        allies = list(visible.allies(observation))
        enemies = list(visible.enemies(observation))

        # ------------------------------------------------------
        # Enemy contact immediately switches to combat.
        # ------------------------------------------------------

        if enemies:
            self.phase = "COMBAT"
            self.has_seen_enemy = True

        # ======================================================
        # COMBAT
        # ======================================================

        if self.phase == "COMBAT" and enemies:

            team_target = self._select_target(
                observation,
                allies,
                enemies,
            )

            if unit_type == "footman":
                return self._footman_tree(
                    observation,
                    allies,
                    enemies,
                    team_target,
                )

            if unit_type == "archer":
                return self._archer_tree(
                    observation,
                    allies,
                    enemies,
                    team_target,
                )

            if unit_type == "cavalry":
                return self._cavalry_tree(
                    observation,
                    allies,
                    enemies,
                    team_target,
                )

            return action.stay()

        # ======================================================
        # COMBAT ENDED / ENEMY LOST
        # ======================================================

        if self.phase == "COMBAT" and not enemies:
            self.phase = "RECOVER"
            self.combat_state = None

        # ======================================================
        # RECOVER
        # ======================================================

        if self.phase == "RECOVER":

            if allies and self._nearest_ally_distance(
                observation,
                allies,
            ) <= self.RALLY_DISTANCE:
                self.phase = "FORM_UP"
            else:
                return self._rally(
                    observation,
                    allies,
                )

        # ======================================================
        # RENDEZVOUS
        # ======================================================

        if self.phase == "RENDEZVOUS":

            if len(allies) >= 2:
                self.phase = "FORM_UP"
            else:
                return self._rendezvous(
                    observation
                )

        # ======================================================
        # FORM UP / PEACETIME ADVANCE
        # ======================================================

        if self.phase == "FORM_UP":

            # Before first contact, all three units try to meet.
            if not self.has_seen_enemy and len(allies) < 2:
                return self._rendezvous(
                    observation
                )

            # After combat, do not wait forever for a dead teammate.
            if allies and not self._formation_ready(
                observation,
                allies,
                unit_type,
            ):
                return self._form_up(
                    observation,
                    allies,
                    unit_type,
                )

            return self._advance_as_team(
                observation,
                allies,
                unit_type,
            )

        return action.stay()

    # ==========================================================
    # STRATEGIC FSM HELPERS
    # ==========================================================

    def _rendezvous(self, observation):

        center = tile.at_center(observation)

        return self._move_toward(
            observation,
            center,
        )

    def _rally(self, observation, allies):

        if not allies:
            return self._rendezvous(
                observation
            )

        footman = self._find_footman(allies)

        if footman is not None:
            goal = footman["position"]
        else:
            here = me.position(observation)

            nearest = min(
                allies,
                key=lambda ally: tile.distance(
                    here,
                    ally["position"],
                ),
            )

            goal = nearest["position"]

        return self._move_toward(
            observation,
            goal,
        )

    def _formation_ready(
        self,
        observation,
        allies,
        unit_type,
    ):

        if unit_type == "footman":
            return True

        footman = self._find_footman(allies)

        if footman is None:
            return False

        here = me.position(observation)

        if unit_type == "archer":
            desired = self._behind_position(
                observation,
                footman["position"],
            )

            return tile.distance(
                here,
                desired,
            ) <= 1

        if unit_type == "cavalry":
            desired = self._cavalry_formation_position(
                observation,
                footman["position"],
            )

            return tile.distance(
                here,
                desired,
            ) <= 1

        return True

    def _form_up(
        self,
        observation,
        allies,
        unit_type,
    ):

        if unit_type == "footman":
            return action.stay()

        footman = self._find_footman(allies)

        if footman is None:
            return self._rally(
                observation,
                allies,
            )

        if unit_type == "archer":
            goal = self._behind_position(
                observation,
                footman["position"],
            )

            return self._move_toward(
                observation,
                goal,
            )

        if unit_type == "cavalry":
            goal = self._cavalry_formation_position(
                observation,
                footman["position"],
            )

            return self._move_toward(
                observation,
                goal,
            )

        return action.stay()

    def _advance_as_team(
        self,
        observation,
        allies,
        unit_type,
    ):

        if unit_type == "footman":
            return self._advance_forward(
                observation
            )

        footman = self._find_footman(allies)

        if footman is None:
            return self._advance_forward(
                observation
            )

        if unit_type == "archer":
            goal = self._behind_position(
                observation,
                footman["position"],
            )

            return self._move_toward(
                observation,
                goal,
            )

        if unit_type == "cavalry":
            goal = self._cavalry_formation_position(
                observation,
                footman["position"],
            )

            return self._move_toward(
                observation,
                goal,
            )

        return action.stay()

    # ==========================================================
    # FOOTMAN BEHAVIOR TREE
    # ==========================================================

    def _footman_tree(
        self,
        observation,
        allies,
        enemies,
        team_target,
    ):

        # ------------------------------------------------------
        # Priority 1:
        # Screen/intercept cavalry threatening our archer.
        # ------------------------------------------------------

        threat_pair = self._cavalry_threat_pair(
            allies,
            enemies,
        )

        if threat_pair is not None:
            threat, archer = threat_pair
            self.combat_state = "SCREEN"

            return self._footman_screen_archer(
                observation,
                threat,
                archer,
            )

        # ------------------------------------------------------
        # Priority 2:
        # Finish an enemy if we can kill it this activation.
        # ------------------------------------------------------

        kill_order = self._reachable_kill_order(
            observation,
            enemies,
            attack_range=self.FOOTMAN_RANGE,
            damage=self.FOOTMAN_DAMAGE,
            prefer_long_move=False,
        )

        if kill_order is not None:
            enemy, path_id = kill_order
            self.combat_state = "FINISH"

            if path_id == 0:
                return action.stay(
                    enemy["unit_id"],
                    observation,
                )

            return action.move(
                path_id,
                enemy["unit_id"],
                observation,
            )

        # ------------------------------------------------------
        # Priority 3:
        # Hold and attack the shared target if adjacent.
        # ------------------------------------------------------

        here = me.position(observation)
        target_distance = tile.distance(
            here,
            team_target["position"],
        )

        if target_distance <= 1:
            self.combat_state = "HOLD"

            return action.stay(
                team_target["unit_id"],
                observation,
            )

        # ------------------------------------------------------
        # Priority 4:
        # Advance and pin the target.
        # ------------------------------------------------------

        self.combat_state = "ADVANCE"

        return self._move_toward(
            observation,
            team_target["position"],
            target_id=team_target["unit_id"],
        )

    def _footman_screen_archer(
        self,
        observation,
        threat,
        archer,
    ):
        """
        Put the footman on the archer-side tile closest to the cavalry.

        This is more useful than simply chasing cavalry because cavalry is
        faster. The footman tries to become a physical obstacle between the
        cavalry and archer.
        """

        here = me.position(observation)

        # If cavalry is already adjacent, hit it.
        if tile.distance(
            here,
            threat["position"],
        ) <= 1:
            return action.stay(
                threat["unit_id"],
                observation,
            )

        screen_position = self._screen_position(
            archer["position"],
            threat["position"],
        )

        # If we are already screening, hold and threaten cavalry.
        if self._same_position(
            here,
            screen_position,
        ):
            return action.stay(
                threat["unit_id"],
                observation,
            )

        return self._move_toward(
            observation,
            screen_position,
            target_id=threat["unit_id"],
        )

    # ==========================================================
    # ARCHER BEHAVIOR TREE
    # ==========================================================

    def _archer_tree(
        self,
        observation,
        allies,
        enemies,
        team_target,
    ):

        here = me.position(observation)

        # ------------------------------------------------------
        # Priority 1:
        # If an enemy is directly adjacent, escape toward the
        # nearest SAFE hill. The move still names that enemy, so
        # the archer will shoot after moving if it remains in range.
        # ------------------------------------------------------

        adjacent_enemies = [
            enemy
            for enemy in enemies
            if tile.distance(
                here,
                enemy["position"],
            ) <= 1
        ]

        if adjacent_enemies:
            immediate_threat = min(
                adjacent_enemies,
                key=lambda enemy: (
                    0 if enemy["type"] == "cavalry" else 1,
                    enemy["hit_points"],
                    enemy["unit_id"],
                ),
            )

            self.combat_state = "ESCAPE_TO_HILL"

            hill_order = self._archer_seek_hill(
                observation,
                allies,
                enemies,
                target=immediate_threat,
                emergency=True,
                require_move=True,
            )

            if hill_order is not None:
                return hill_order

            # If no suitable nearby hill exists, preserve the old survival
            # behavior rather than walking toward a dangerous hill.
            return self._archer_kite(
                observation,
                allies,
                enemies,
                immediate_threat,
            )

        # ------------------------------------------------------
        # Priority 2:
        # Visible cavalry remains urgent even before it reaches
        # melee range. Prefer escaping toward a safe hill; fall
        # back to the normal kite utility if no hill is suitable.
        # ------------------------------------------------------

        threat = self._archer_threat(
            observation,
            enemies,
        )

        if threat is not None:
            self.combat_state = "HILL_KITE"

            hill_order = self._archer_seek_hill(
                observation,
                allies,
                enemies,
                target=threat,
                emergency=True,
                require_move=False,
            )

            if hill_order is not None:
                return hill_order

            return self._archer_kite(
                observation,
                allies,
                enemies,
                threat,
            )

        # ------------------------------------------------------
        # Priority 3:
        # Once combat begins, establish a nearby hill firing
        # position before normal target selection. If already on
        # a hill, this returns None and the archer starts shooting.
        # ------------------------------------------------------

        if not self._is_hill(
            observation,
            here,
        ):
            self.combat_state = "TAKE_HILL"

            hill_order = self._archer_seek_hill(
                observation,
                allies,
                enemies,
                target=team_target,
                emergency=False,
                require_move=False,
            )

            if hill_order is not None:
                return hill_order

        # ------------------------------------------------------
        # Priority 4:
        # Guaranteed kill from the current firing position.
        # ------------------------------------------------------

        kill_target = self._archer_kill_target(
            observation,
            enemies,
        )

        if kill_target is not None:
            self.combat_state = "FINISH"

            return action.stay(
                kill_target["unit_id"],
                observation,
            )

        # ------------------------------------------------------
        # Priority 5:
        # Fire the team target if it is in range.
        # ------------------------------------------------------

        target_distance = tile.distance(
            here,
            team_target["position"],
        )

        if target_distance <= self.ARCHER_RANGE:
            self.combat_state = "FIRE_FROM_HILL"

            return action.stay(
                team_target["unit_id"],
                observation,
            )

        # ------------------------------------------------------
        # Priority 6:
        # No nearby hill or useful shot: use the existing utility
        # repositioning logic.
        # ------------------------------------------------------

        self.combat_state = "REPOSITION"

        return self._archer_reposition(
            observation,
            allies,
            enemies,
            team_target,
        )

    def _archer_seek_hill(
        self,
        observation,
        allies,
        enemies,
        target,
        emergency,
        require_move,
    ):
        """
        Move toward the nearest tactically acceptable hill.

        "Nearest" is measured by actual terrain route cost, not hex-line
        distance, so water barriers are handled correctly. In emergencies we
        reject obviously dangerous hills before considering route length.

        The returned move names ``target``. Because Skirmish resolves movement
        first and then attacks, the archer can reach/approach the hill and still
        fire at that enemy if it remains in range from the final tile.
        """

        here = me.position(observation)

        if (
            not require_move
            and self._is_hill(observation, here)
        ):
            return None

        hill = self._best_nearby_hill(
            observation,
            allies,
            enemies,
            emergency=emergency,
            exclude_current=require_move,
        )

        if hill is None:
            return None

        target_id = (
            target["unit_id"]
            if target is not None
            else None
        )

        return self._move_toward(
            observation,
            hill,
            target_id=target_id,
        )

    def _best_nearby_hill(
        self,
        observation,
        allies,
        enemies,
        emergency,
        exclude_current,
    ):
        """
        Choose a nearby hill for the archer.

        Route cost is the primary notion of nearness. Safety breaks ties and,
        during an emergency, unsafe hills are placed behind safe hills even if
        they are geometrically closer. Visible occupied hills are skipped.
        """

        here = me.position(observation)
        footman = self._find_footman(allies)
        occupied = {
            self._position_key(unit["position"])
            for unit in [*allies, *enemies]
        }

        max_cost = (
            self.ARCHER_EMERGENCY_HILL_SEARCH_COST
            if emergency
            else self.ARCHER_HILL_SEARCH_COST
        )

        best_hill = None
        best_key = None

        for hill in self._all_hills(observation):
            if (
                exclude_current
                and self._same_position(hill, here)
            ):
                continue

            if (
                not self._same_position(hill, here)
                and self._position_key(hill) in occupied
            ):
                continue

            route_costs = self._route_cost_map(
                observation,
                hill,
            )
            route_cost = route_costs.get(
                self._position_key(here),
                float("inf"),
            )

            if route_cost == float("inf"):
                continue

            if route_cost > max_cost:
                continue

            if enemies:
                nearest_enemy_distance = min(
                    tile.distance(
                        hill,
                        enemy["position"],
                    )
                    for enemy in enemies
                )
            else:
                nearest_enemy_distance = 99

            cavalry_distances = [
                tile.distance(
                    hill,
                    enemy["position"],
                )
                for enemy in enemies
                if enemy["type"] == "cavalry"
            ]
            nearest_cavalry_distance = (
                min(cavalry_distances)
                if cavalry_distances
                else 99
            )

            # Emergency hills should actually create separation. An unacted
            # cavalry within four tiles of the hill is especially dangerous.
            emergency_danger = 0
            if emergency:
                if nearest_enemy_distance <= 1:
                    emergency_danger += 3
                elif nearest_enemy_distance <= 2:
                    emergency_danger += 1

                for enemy in enemies:
                    if enemy["type"] != "cavalry":
                        continue

                    cavalry_distance = tile.distance(
                        hill,
                        enemy["position"],
                    )

                    if (
                        not self._has_acted(enemy)
                        and cavalry_distance <= 4
                    ):
                        emergency_danger += 3
                    elif cavalry_distance <= 3:
                        emergency_danger += 2

            support_distance = 99
            if footman is not None:
                support_distance = tile.distance(
                    hill,
                    footman["position"],
                )

            # Prefer hills that can still attack at least one visible enemy.
            firing_penalty = 1
            if any(
                tile.distance(
                    hill,
                    enemy["position"],
                ) <= self.ARCHER_RANGE
                for enemy in enemies
            ):
                firing_penalty = 0

            # Route cost determines "nearby". During emergencies, safety is
            # considered before route cost so the archer does not flee onto a
            # hill that remains inside immediate melee/cavalry reach.
            if emergency:
                key = (
                    emergency_danger,
                    route_cost,
                    -nearest_cavalry_distance,
                    -nearest_enemy_distance,
                    firing_penalty,
                    support_distance,
                    hill["q"],
                    hill["r"],
                )
            else:
                key = (
                    route_cost,
                    firing_penalty,
                    -nearest_enemy_distance,
                    support_distance,
                    hill["q"],
                    hill["r"],
                )

            if best_key is None or key < best_key:
                best_key = key
                best_hill = hill

        return best_hill

    def _all_hills(self, observation):
        """Return every hill position from the shared Season 2 battlefield."""

        raw = observation.get(
            "observation",
            observation,
        )
        battlefield = raw.get(
            "battlefield",
            {},
        )
        tiles_grid = battlefield.get(
            "tiles",
            [],
        )

        hills = []

        for r, row in enumerate(tiles_grid):
            for q, info in enumerate(row):
                if not isinstance(info, dict):
                    continue

                if info.get("terrain") == "hill":
                    hills.append(
                        {"q": q, "r": r}
                    )

        return hills

    def _is_hill(
        self,
        observation,
        position,
    ):
        info = tile.terrain_at(
            observation,
            position,
        )

        return bool(
            info
            and info.get("terrain") == "hill"
        )

    def _archer_threat(
        self,
        observation,
        enemies,
    ):
        """
        Return the most urgent enemy threatening the archer.

        Season 2 observations expose has_acted. An unacted cavalry can still
        move this round, so it is urgent anywhere in the archer's 6-tile vision.
        A cavalry that already acted is still dangerous near the round boundary,
        but we allow a slightly smaller immediate-danger radius.
        """

        here = me.position(observation)
        best_threat = None
        best_score = float("-inf")

        for enemy in enemies:
            distance = tile.distance(
                here,
                enemy["position"],
            )

            has_acted = self._has_acted(enemy)

            if enemy["type"] == "cavalry":
                danger_radius = 6 if not has_acted else 5

                if distance > danger_radius:
                    continue

                closeness = 1.0 - min(1.0, distance / 6.0)
                activation_risk = 1.0 if not has_acted else 0.65

                score = (
                    0.65 * (closeness ** 2)
                    + 0.35 * activation_risk
                )

            else:
                if distance > self.ARCHER_MELEE_DANGER:
                    continue

                closeness = 1.0 - min(
                    1.0,
                    distance / float(self.ARCHER_MELEE_DANGER + 1),
                )
                activation_risk = 1.0 if not has_acted else 0.5

                score = (
                    0.75 * closeness
                    + 0.25 * activation_risk
                )

            if score > best_score:
                best_score = score
                best_threat = enemy

        return best_threat

    def _archer_kill_target(
        self,
        observation,
        enemies,
    ):

        here = me.position(observation)

        candidates = [
            enemy
            for enemy in enemies
            if (
                enemy["hit_points"]
                <= self.ARCHER_DAMAGE
                and tile.distance(
                    here,
                    enemy["position"],
                )
                <= self.ARCHER_RANGE
            )
        ]

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda enemy: (
                enemy["hit_points"],
                tile.distance(
                    here,
                    enemy["position"],
                ),
                enemy["unit_id"],
            ),
        )

    def _archer_kite(
        self,
        observation,
        allies,
        enemies,
        threat,
    ):
        """
        Utility-based escape.

        The behavior tree has already decided that survival is the objective.
        Utility AI now scores every legal path using:
            * threat distance,
            * whether the threat has acted,
            * safety from other enemies,
            * footman screening/support,
            * Season 2 terrain,
            * center/edge safety,
            * use of the archer's available movement.
        """

        here = me.position(observation)
        center = tile.at_center(observation)
        footman = self._find_footman(allies)
        threat_has_acted = self._has_acted(threat)

        best_path = 0
        best_score = float("-inf")

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(
                here,
                path_id,
            )

            step_count = len(paths.decode(path_id))
            threat_distance = tile.distance(
                destination,
                threat["position"],
            )

            # ------------------------------
            # 1. Threat safety, normalized.
            # ------------------------------
            if threat["type"] == "cavalry":
                safety_reference = 7.0
                threat_safety = min(
                    1.0,
                    threat_distance / safety_reference,
                )

                # Unacted cavalry is more urgent because it can still move in
                # the current round. Acted cavalry remains risky near a round
                # boundary, so it is not treated as harmless.
                activation_factor = 1.0 if not threat_has_acted else 0.75
                threat_safety *= activation_factor
            else:
                threat_safety = min(
                    1.0,
                    threat_distance / 4.0,
                )

            # ------------------------------
            # 2. Safety from all other enemies.
            # ------------------------------
            other_safety_values = []

            for enemy in enemies:
                if enemy["unit_id"] == threat["unit_id"]:
                    continue

                enemy_distance = tile.distance(
                    destination,
                    enemy["position"],
                )

                reference = 6.0 if enemy["type"] == "cavalry" else 4.0
                other_safety_values.append(
                    min(1.0, enemy_distance / reference)
                )

            other_safety = (
                min(other_safety_values)
                if other_safety_values
                else 1.0
            )

            # ------------------------------
            # 3. Footman support / screening.
            # ------------------------------
            support = 0.5
            screen = 0.0

            if footman is not None:
                footman_position = footman["position"]
                footman_distance = tile.distance(
                    destination,
                    footman_position,
                )

                support = 1.0 - min(
                    1.0,
                    footman_distance / 6.0,
                )

                threat_to_footman = tile.distance(
                    threat["position"],
                    footman_position,
                )

                if threat_to_footman < threat_distance:
                    screen += 0.45

                if (
                    threat_to_footman
                    + footman_distance
                    <= threat_distance + 1
                ):
                    screen += 0.55

                screen = min(1.0, screen)

            # ------------------------------
            # 4. Terrain and board-position utility.
            # ------------------------------
            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="archer",
                purpose="defense",
            )

            center_utility = 1.0 - min(
                1.0,
                tile.distance(destination, center) / 8.0,
            )

            movement_utility = min(
                1.0,
                step_count / 2.0,
            )

            score = (
                0.38 * threat_safety
                + 0.14 * other_safety
                + 0.16 * support
                + 0.12 * screen
                + 0.10 * terrain_utility
                + 0.05 * center_utility
                + 0.05 * movement_utility
            )

            # Hard safety corrections. Utility is useful, but the archer should
            # not choose a beautiful hill if cavalry can immediately reach it.
            if threat["type"] == "cavalry":
                unsafe_limit = 5 if not threat_has_acted else 4
                if threat_distance <= unsafe_limit:
                    score -= 1.0

            if any(
                tile.distance(destination, enemy["position"]) <= 1
                for enemy in enemies
                if enemy["unit_id"] != threat["unit_id"]
            ):
                score -= 1.0

            if score > best_score:
                best_score = score
                best_path = path_id

        if best_path != 0:
            return action.move(
                best_path,
                threat["unit_id"],
                observation,
            )

        return action.stay(
            threat["unit_id"],
            observation,
        )

    def _archer_reposition(
        self,
        observation,
        allies,
        enemies,
        target,
    ):
        """Utility-based archer positioning for Season 2 terrain."""

        here = me.position(observation)
        center = tile.at_center(observation)
        footman = self._find_footman(allies)

        best_path = 0
        best_score = float("-inf")

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(
                here,
                path_id,
            )

            target_distance = tile.distance(
                destination,
                target["position"],
            )

            # Ideal-range utility: 1.0 at ideal distance and falls away from it.
            range_utility = 1.0 - min(
                1.0,
                abs(target_distance - self.ARCHER_IDEAL_DISTANCE)
                / float(self.ARCHER_IDEAL_DISTANCE),
            )

            # Safety utility uses the most dangerous visible enemy after moving.
            safety_values = []

            for enemy in enemies:
                enemy_distance = tile.distance(
                    destination,
                    enemy["position"],
                )

                if enemy["type"] == "cavalry":
                    base = min(1.0, enemy_distance / 7.0)
                    if not self._has_acted(enemy):
                        base *= 0.8
                else:
                    base = min(1.0, enemy_distance / 4.0)

                safety_values.append(base)

            safety_utility = (
                min(safety_values)
                if safety_values
                else 1.0
            )

            support_utility = 0.5

            if footman is not None:
                footman_distance = tile.distance(
                    destination,
                    footman["position"],
                )

                support_utility = 1.0 - min(
                    1.0,
                    footman_distance / 6.0,
                )

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="archer",
                purpose="attack",
            )

            center_utility = 1.0 - min(
                1.0,
                tile.distance(destination, center) / 8.0,
            )

            score = (
                0.32 * range_utility
                + 0.30 * safety_utility
                + 0.16 * support_utility
                + 0.16 * terrain_utility
                + 0.06 * center_utility
            )

            # Do not step into immediate melee just to gain terrain/range value.
            if any(
                tile.distance(destination, enemy["position"]) <= 1
                for enemy in enemies
            ):
                score -= 1.0

            if score > best_score:
                best_score = score
                best_path = path_id

        if best_path == 0:
            return action.stay(
                target["unit_id"],
                observation,
            )

        return action.move(
            best_path,
            target["unit_id"],
            observation,
        )

    def _cavalry_tree(
        self,
        observation,
        allies,
        enemies,
        team_target,
    ):

        footman = self._find_footman(allies)

        # ------------------------------------------------------
        # Priority 1:
        # After a charge, disengage before anything else.
        # ------------------------------------------------------

        if self.combat_state == "CHARGE":

            self.combat_state = "DISENGAGE"

            return self._cavalry_disengage(
                observation,
                enemies,
                footman,
            )

        # ------------------------------------------------------
        # Priority 2:
        # Counter enemy cavalry threatening our archer.
        # ------------------------------------------------------

        threat_pair = self._cavalry_threat_pair(
            allies,
            enemies,
        )

        if threat_pair is not None:
            threat, _archer = threat_pair
            self.combat_state = "COUNTER_CAVALRY"

            return self._cavalry_pressure_target(
                observation,
                threat,
                enemies,
                footman,
            )

        # ------------------------------------------------------
        # Priority 3:
        # Reachable kill.
        # ------------------------------------------------------

        kill_order = self._reachable_kill_order(
            observation,
            enemies,
            attack_range=self.CAVALRY_RANGE,
            damage=self.CAVALRY_DAMAGE,
            prefer_long_move=False,
        )

        if kill_order is not None:

            enemy, path_id = kill_order
            self.combat_state = "CHARGE"

            if path_id == 0:
                return action.stay(
                    enemy["unit_id"],
                    observation,
                )

            return action.move(
                path_id,
                enemy["unit_id"],
                observation,
            )

        # ------------------------------------------------------
        # Priority 4:
        # If already adjacent, attack now and disengage next time.
        # ------------------------------------------------------

        here = me.position(observation)

        adjacent = [
            enemy
            for enemy in enemies
            if tile.distance(
                here,
                enemy["position"],
            ) <= 1
        ]

        if adjacent:

            enemy = min(
                adjacent,
                key=lambda item: (
                    item["hit_points"],
                    item["unit_id"],
                ),
            )

            self.combat_state = "CHARGE"

            return action.stay(
                enemy["unit_id"],
                observation,
            )

        # ------------------------------------------------------
        # Priority 5:
        # Normal flank/charge only after footman pins target.
        # ------------------------------------------------------

        if footman is not None:

            footman_distance = tile.distance(
                footman["position"],
                team_target["position"],
            )

            if footman_distance <= 1:

                self.combat_state = "FLANK"

                return self._cavalry_flank_or_charge(
                    observation,
                    team_target,
                    footman,
                    enemies,
                )

        # ------------------------------------------------------
        # Priority 6:
        # Wait beside / behind footman.
        # ------------------------------------------------------

        self.combat_state = "WAIT"

        if footman is not None:

            goal = self._cavalry_formation_position(
                observation,
                footman["position"],
            )

            return self._move_toward(
                observation,
                goal,
            )

        return action.stay()

    def _cavalry_pressure_target(
        self,
        observation,
        target,
        enemies,
        footman,
    ):
        """Utility AI for counter-pressuring enemy cavalry in Season 2."""

        here = me.position(observation)
        center = tile.at_center(observation)
        best_path = 0
        best_score = float("-inf")
        best_final_distance = None

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(
                here,
                path_id,
            )

            target_distance = tile.distance(
                destination,
                target["position"],
            )

            access_utility = 1.0 - min(
                1.0,
                max(0, target_distance - 1) / 6.0,
            )

            nearby_enemies = sum(
                1
                for enemy in enemies
                if tile.distance(
                    destination,
                    enemy["position"],
                ) <= 1
            )

            safety_utility = 1.0 - min(
                1.0,
                nearby_enemies / 3.0,
            )

            support_utility = 0.5
            if footman is not None:
                support_distance = tile.distance(
                    destination,
                    footman["position"],
                )
                support_utility = 1.0 - min(
                    1.0,
                    support_distance / 7.0,
                )

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="cavalry",
                purpose="mobility",
            )

            center_utility = 1.0 - min(
                1.0,
                tile.distance(destination, center) / 8.0,
            )

            score = (
                0.42 * access_utility
                + 0.25 * safety_utility
                + 0.13 * support_utility
                + 0.15 * terrain_utility
                + 0.05 * center_utility
            )

            if nearby_enemies >= 2:
                score -= 0.75

            if score > best_score:
                best_score = score
                best_path = path_id
                best_final_distance = target_distance

        if best_path == 0:
            return action.stay()

        if best_final_distance == 1:
            self.combat_state = "CHARGE"
            return action.move(
                best_path,
                target["unit_id"],
                observation,
            )

        return action.move(best_path)

    def _cavalry_flank_or_charge(
        self,
        observation,
        target,
        footman,
        enemies,
    ):
        """
        Utility-based Season 2 cavalry flank selection.

        Season 2 has abilities disabled, so path length receives no charge-damage
        bonus. The cavalry still exploits its 4 movement points to reach strong
        flank positions, but utility focuses on access, flank angle, survival,
        terrain, support, and not blocking the footman.
        """

        here = me.position(observation)
        center = tile.at_center(observation)
        footman_position = footman["position"]
        front_of_footman = self._front_position(
            observation,
            footman_position,
        )

        best_path = 0
        best_score = float("-inf")
        best_final_distance = None

        for path_id in action.legal_paths(observation):
            if path_id == 0:
                continue

            destination = tile.at_path_end(
                here,
                path_id,
            )

            target_distance = tile.distance(
                destination,
                target["position"],
            )

            access_utility = 1.0 - min(
                1.0,
                max(0, target_distance - 1) / 6.0,
            )

            flank_offset = self._flank_offset(
                footman_position,
                target["position"],
                destination,
            )
            flank_utility = min(1.0, flank_offset / 5.0)

            nearby_enemies = sum(
                1
                for enemy in enemies
                if tile.distance(
                    destination,
                    enemy["position"],
                ) <= 1
            )
            safety_utility = 1.0 - min(
                1.0,
                nearby_enemies / 3.0,
            )

            footman_distance = tile.distance(
                destination,
                footman_position,
            )
            support_utility = 1.0 - min(
                1.0,
                abs(footman_distance - 3) / 5.0,
            )

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="cavalry",
                purpose="mobility",
            )

            center_utility = 1.0 - min(
                1.0,
                tile.distance(destination, center) / 8.0,
            )

            score = (
                0.33 * access_utility
                + 0.23 * flank_utility
                + 0.22 * safety_utility
                + 0.08 * support_utility
                + 0.10 * terrain_utility
                + 0.04 * center_utility
            )

            if self._same_position(
                destination,
                front_of_footman,
            ):
                score -= 1.5

            if nearby_enemies >= 2:
                score -= 0.8

            if score > best_score:
                best_score = score
                best_path = path_id
                best_final_distance = target_distance

        if best_path == 0:
            return action.stay()

        if best_final_distance == 1:
            self.combat_state = "CHARGE"
            return action.move(
                best_path,
                target["unit_id"],
                observation,
            )

        self.combat_state = "FLANK"
        return action.move(best_path)

    def _cavalry_disengage(
        self,
        observation,
        enemies,
        footman,
    ):
        """Utility-based disengage that avoids marsh and enemy clusters."""

        here = me.position(observation)
        center = tile.at_center(observation)

        best_path = 0
        best_score = float("-inf")

        for path_id in action.legal_paths(observation):
            if path_id == 0:
                continue

            destination = tile.at_path_end(
                here,
                path_id,
            )

            nearest_enemy_distance = min(
                tile.distance(
                    destination,
                    enemy["position"],
                )
                for enemy in enemies
            )

            reset_utility = 1.0 - min(
                1.0,
                abs(
                    nearest_enemy_distance
                    - self.CAVALRY_RESET_DISTANCE
                ) / 4.0,
            )

            safety_utility = min(
                1.0,
                nearest_enemy_distance / 4.0,
            )

            support_utility = 0.5
            if footman is not None:
                footman_distance = tile.distance(
                    destination,
                    footman["position"],
                )
                support_utility = 1.0 - min(
                    1.0,
                    abs(footman_distance - 3) / 5.0,
                )

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="cavalry",
                purpose="mobility",
            )

            center_utility = 1.0 - min(
                1.0,
                tile.distance(destination, center) / 8.0,
            )

            score = (
                0.34 * reset_utility
                + 0.30 * safety_utility
                + 0.14 * support_utility
                + 0.16 * terrain_utility
                + 0.06 * center_utility
            )

            if nearest_enemy_distance <= 1:
                score -= 1.0

            if score > best_score:
                best_score = score
                best_path = path_id

        self.combat_state = "FLANK"

        if best_path != 0:
            return action.move(best_path)

        return action.stay()

    def _select_target(
        self,
        observation,
        allies,
        enemies,
    ):
        """
        Utility-AI target selection.

        Each enemy receives normalized scores for:
            closeness,
            weakness,
            type threat,
            activation threat (has_acted),
            danger to a visible allied archer,
            whether this unit can kill it now,
            and commitment to the current target.
        """

        if not enemies:
            self.current_target_id = None
            return None

        best_enemy = None
        best_score = float("-inf")

        for enemy in enemies:
            score = self._score_enemy_utility(
                observation,
                allies,
                enemy,
            )

            if score > best_score:
                best_score = score
                best_enemy = enemy

        if best_enemy is not None:
            self.current_target_id = best_enemy["unit_id"]

        return best_enemy

    def _cavalry_threat_pair(
        self,
        allies,
        enemies,
    ):
        """
        Return (enemy_cavalry, allied_archer) for the most urgent visible
        cavalry threat, or None.
        """

        archers = [
            ally
            for ally in allies
            if ally["type"] == "archer"
        ]

        if not archers:
            return None

        candidates = []

        for enemy in enemies:

            if enemy["type"] != "cavalry":
                continue

            for archer in archers:

                distance = tile.distance(
                    enemy["position"],
                    archer["position"],
                )

                if distance <= self.ARCHER_CAVALRY_THREAT:
                    candidates.append(
                        (
                            distance,
                            enemy["hit_points"],
                            enemy["unit_id"],
                            enemy,
                            archer,
                        )
                    )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        best = candidates[0]

        return best[3], best[4]

    def _reachable_kill_order(
        self,
        observation,
        enemies,
        attack_range,
        damage,
        prefer_long_move,
    ):

        here = me.position(observation)
        legal_paths = list(
            action.legal_paths(observation)
        )

        best = None
        best_key = None

        for enemy in enemies:

            if enemy["hit_points"] > damage:
                continue

            for path_id in legal_paths:

                destination = tile.at_path_end(
                    here,
                    path_id,
                )

                final_distance = tile.distance(
                    destination,
                    enemy["position"],
                )

                if final_distance > attack_range:
                    continue

                step_count = len(
                    paths.decode(path_id)
                )

                if prefer_long_move:
                    long_move_penalty = (
                        0
                        if step_count >= 3
                        else 1
                    )
                else:
                    long_move_penalty = 0

                key = (
                    long_move_penalty,
                    enemy["hit_points"],
                    step_count,
                    enemy["unit_id"],
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best = (
                        enemy,
                        path_id,
                    )

        return best

    # ==========================================================
    # UTILITY AI HELPERS
    # ==========================================================

    def _score_enemy_utility(
        self,
        observation,
        allies,
        enemy,
    ):
        here = me.position(observation)
        unit_type = me.unit_type(observation)

        distance = tile.distance(
            here,
            enemy["position"],
        )

        closeness = 1.0 - min(
            1.0,
            distance / 10.0,
        )

        max_hp = {
            "footman": 12.0,
            "archer": 6.0,
            "cavalry": 10.0,
        }.get(enemy["type"], 12.0)

        weakness = 1.0 - min(
            1.0,
            enemy["hit_points"] / max_hp,
        )

        type_threat = {
            "cavalry": 1.0,
            "archer": 0.7,
            "footman": 0.5,
        }.get(enemy["type"], 0.5)

        activation_threat = (
            1.0
            if not self._has_acted(enemy)
            else 0.45
        )

        # Threat to an allied archer is a shared team concern.
        archer_threat = 0.0
        allied_archers = [
            ally
            for ally in allies
            if ally["type"] == "archer"
        ]

        if enemy["type"] == "cavalry" and allied_archers:
            nearest_archer_distance = min(
                tile.distance(
                    enemy["position"],
                    archer["position"],
                )
                for archer in allied_archers
            )
            archer_threat = 1.0 - min(
                1.0,
                nearest_archer_distance / 6.0,
            )

        damage = {
            "footman": self.FOOTMAN_DAMAGE,
            "archer": self.ARCHER_DAMAGE,
            "cavalry": self.CAVALRY_DAMAGE,
        }.get(unit_type, 2)

        attack_range = {
            "footman": self.FOOTMAN_RANGE,
            "archer": self.ARCHER_RANGE,
            "cavalry": self.CAVALRY_RANGE,
        }.get(unit_type, 1)

        killable_now = (
            1.0
            if (
                enemy["hit_points"] <= damage
                and distance <= attack_range
            )
            else 0.0
        )

        in_range = (
            1.0
            if distance <= attack_range
            else 0.0
        )

        commitment = (
            1.0
            if enemy["unit_id"] == self.current_target_id
            else 0.0
        )

        # Shared weights keep different unit instances more likely to focus the
        # same target. Survival-specific exceptions are handled by the behavior
        # tree before this target chooser runs.
        score = (
            0.22 * closeness
            + 0.22 * weakness
            + 0.17 * type_threat
            + 0.10 * activation_threat
            + 0.11 * archer_threat
            + 0.08 * in_range
            + 0.10 * killable_now
        )

        # Small persistence bonus prevents target oscillation when utilities are
        # almost tied.
        score += 0.10 * commitment

        return score

    def _terrain_utility(
        self,
        observation,
        position,
        unit_type,
        purpose,
    ):
        """
        Return a normalized-ish 0..1 preference for Season 2 terrain.

        Hills are valuable firing/holding ground. Forest is especially useful
        for an archer under ranged fire. Marsh is undesirable because it slows
        movement. Water never appears as a legal destination, but is treated as
        zero utility defensively.
        """

        info = tile.terrain_at(
            observation,
            position,
        )

        if not info:
            return 0.0

        terrain = info.get("terrain")
        feature = info.get("feature")

        score = 0.50

        if terrain == "hill":
            if unit_type == "archer":
                score += 0.30
            elif unit_type == "footman":
                score += 0.22
            else:
                score += 0.10

        if terrain == "water" or terrain == "void":
            return 0.0

        if feature == "forest":
            if unit_type == "archer":
                score += 0.22
            elif unit_type == "footman":
                score += 0.10
            else:
                score += 0.04

        if feature == "marsh":
            if unit_type == "cavalry":
                score -= 0.35
            elif unit_type == "archer":
                score -= 0.22
            else:
                score -= 0.18

        # In a mobility decision, open ground is slightly preferred because the
        # unit is less likely to lose tempo on the following activation.
        if purpose == "mobility" and feature == "none":
            score += 0.08

        return self._clamp01(score)

    def _has_acted(self, unit):
        return bool(unit.get("has_acted", False))

    def _clamp01(self, value):
        return max(0.0, min(1.0, value))

    # ==========================================================
    # FORMATION / GEOMETRY HELPERS
    # ==========================================================

    def _find_footman(self, allies):

        for ally in allies:
            if ally["type"] == "footman":
                return ally

        return None

    def _nearest_ally_distance(
        self,
        observation,
        allies,
    ):

        if not allies:
            return 999

        here = me.position(observation)

        return min(
            tile.distance(
                here,
                ally["position"],
            )
            for ally in allies
        )

    def _behind_position(
        self,
        observation,
        footman_position,
    ):

        forward = me.direction(observation)

        if forward == 2:
            direction = 5
        else:
            direction = 2

        return tile.neighbors(
            footman_position
        )[direction]

    def _behind_ally_from_threat(
        self,
        ally_position,
        threat_position,
    ):
        """
        Return the neighbor of ally_position farthest from threat_position.
        This gives the archer a threat-relative 'behind the footman' tile.
        """

        neighbors = tile.neighbors(
            ally_position
        )

        return max(
            neighbors.values(),
            key=lambda position: tile.distance(
                position,
                threat_position,
            ),
        )

    def _screen_position(
        self,
        protected_position,
        threat_position,
    ):
        """
        Return the neighbor of the protected unit closest to the threat.
        The footman tries to occupy this tile as a physical screen.
        """

        neighbors = tile.neighbors(
            protected_position
        )

        return min(
            neighbors.values(),
            key=lambda position: tile.distance(
                position,
                threat_position,
            ),
        )

    def _cavalry_formation_position(
        self,
        observation,
        footman_position,
    ):

        forward = me.direction(observation)

        if forward == 2:
            direction = 6
        else:
            direction = 3

        return tile.neighbors(
            footman_position
        )[direction]

    def _front_position(
        self,
        observation,
        footman_position,
    ):

        forward = me.direction(observation)

        return tile.neighbors(
            footman_position
        )[forward]

    def _flank_offset(
        self,
        friendly_position,
        enemy_position,
        moving_position,
    ):

        fq = friendly_position["q"]
        fr = friendly_position["r"]

        eq = enemy_position["q"]
        er = enemy_position["r"]

        mq = moving_position["q"]
        mr = moving_position["r"]

        line_q = eq - fq
        line_r = er - fr

        move_q = mq - eq
        move_r = mr - er

        return abs(
            line_q * move_r
            - line_r * move_q
        )

    def _same_position(
        self,
        first,
        second,
    ):

        return (
            first["q"] == second["q"]
            and first["r"] == second["r"]
        )

    # ==========================================================
    # MOVEMENT HELPERS
    # ==========================================================

    def _move_toward(
        self,
        observation,
        goal: AxialPosition,
        target_id=None,
    ):
        """
        Move toward a goal using terrain-aware strategic pathfinding.

        The old implementation minimized straight-line hex distance. That
        fails at water barriers because the correct route may have to move
        sideways, or even briefly farther from the goal, to reach a passage.

        We now build a reverse Dijkstra cost map from the goal across all
        passable terrain. Among the paths that are legal THIS activation, we
        choose the endpoint with the smallest remaining route cost. The action
        mask still remains authoritative for the actual move.
        """

        here = me.position(observation)
        route_costs = self._route_cost_map(
            observation,
            goal,
        )

        best_path = 0
        best_key = None

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(
                here,
                path_id,
            )

            destination_key = self._position_key(
                destination
            )
            route_cost = route_costs.get(
                destination_key,
                float("inf"),
            )

            # Prefer real terrain-route progress first. Straight-line distance
            # is only a tie-breaker now, so water can no longer create a
            # permanent stand-off. If two endpoints are otherwise equivalent,
            # use more of the current activation's legal movement.
            direct_distance = tile.distance(
                destination,
                goal,
            )
            step_count = len(paths.decode(path_id))

            key = (
                route_cost,
                direct_distance,
                -step_count,
                path_id,
            )

            if best_key is None or key < best_key:
                best_key = key
                best_path = path_id

        # If pathfinding cannot find any strategic route, fall back to stay.
        # In the documented battlefield every passable tile is connected, so
        # this should mainly protect against an impassable/void goal.
        if best_key is None or best_key[0] == float("inf"):
            if target_id is not None:
                return action.stay(
                    target_id,
                    observation,
                )
            return action.stay()

        if best_path == 0:
            if target_id is not None:
                return action.stay(
                    target_id,
                    observation,
                )
            return action.stay()

        if target_id is not None:
            return action.move(
                best_path,
                target_id,
                observation,
            )

        return action.move(best_path)

    def _route_cost_map(
        self,
        observation,
        goal: AxialPosition,
    ):
        """
        Return minimum remaining movement cost from every passable tile to goal.

        This is reverse Dijkstra. Terrain costs are paid when ENTERING a tile,
        so while expanding backward from ``current`` to a predecessor we add
        the entry cost of ``current``. Unit occupancy is deliberately ignored
        here because this is multi-turn strategic routing; the current action
        mask handles units and other immediate legality constraints.
        """

        goals = self._strategic_goal_tiles(
            observation,
            goal,
        )

        costs = {}
        frontier = []
        serial = 0

        for goal_tile in goals:
            key = self._position_key(goal_tile)
            if key in costs:
                continue
            costs[key] = 0.0
            heapq.heappush(
                frontier,
                (0.0, serial, goal_tile),
            )
            serial += 1

        while frontier:
            current_cost, _serial, current = heapq.heappop(
                frontier
            )
            current_key = self._position_key(current)

            if current_cost != costs.get(current_key):
                continue

            enter_current = self._strategic_move_cost(
                observation,
                current,
            )

            for predecessor in tile.neighbors(current).values():
                if not self._strategically_passable(
                    observation,
                    predecessor,
                ):
                    continue

                predecessor_key = self._position_key(
                    predecessor
                )
                new_cost = current_cost + enter_current

                if new_cost < costs.get(
                    predecessor_key,
                    float("inf"),
                ):
                    costs[predecessor_key] = new_cost
                    heapq.heappush(
                        frontier,
                        (new_cost, serial, predecessor),
                    )
                    serial += 1

        return costs

    def _strategic_goal_tiles(
        self,
        observation,
        goal: AxialPosition,
    ):
        """Use the goal itself, or its passable neighbors if goal is blocked."""

        if self._strategically_passable(
            observation,
            goal,
        ):
            return [goal]

        alternatives = []
        for neighbor in tile.neighbors(goal).values():
            if self._strategically_passable(
                observation,
                neighbor,
            ):
                alternatives.append(neighbor)

        return alternatives

    def _strategically_passable(
        self,
        observation,
        position,
    ):
        info = tile.terrain_at(
            observation,
            position,
        )

        if not info:
            return False

        terrain = info.get("terrain")
        return terrain not in ("water", "void")

    def _strategic_move_cost(
        self,
        observation,
        position,
    ):
        """Movement cost used only for strategic route selection."""

        info = tile.terrain_at(
            observation,
            position,
        )

        if not info:
            return float("inf")

        terrain = info.get("terrain")
        feature = info.get("feature")

        if terrain in ("water", "void"):
            return float("inf")

        if terrain == "hill":
            cost = 2.0
        else:
            cost = 1.0

        if feature == "forest":
            cost += 1.0
        elif feature == "marsh":
            cost += 2.0

        return cost

    def _position_key(
        self,
        position,
    ):
        return (
            position["q"],
            position["r"],
        )

    def _advance_forward(
        self,
        observation,
    ):

        here = me.position(observation)

        goal = tile.at_mirror(
            here,
            observation,
        )

        return self._move_toward(
            observation,
            goal,
        )

class Season3Agent(Season2CoreAgent):
    """
    Season 3 v5: active 20-unit squads + messaging + Shield Wall + Charge.

    This class reuses the proven Season 2 tactical core (FSM, behavior trees,
    Utility AI, hill-seeking archers, and terrain-aware Dijkstra routing), but
    replaces the three-unit formation assumptions with deterministic squads and
    adds direct peer communication.

    Army plan (20 units):
        Squad 0: footman 0, footman 1, archer 0, cavalry 0
        Squad 1: footman 2, footman 3, archer 1, cavalry 1
        Squad 2: footman 4, archer 2, cavalry 2
        Squad 3: footman 5, archer 3, cavalry 3
        Squad 4: footman 6, archer 4, cavalry 4
        Squad 5: footman 7, archer 5, cavalry 5

    Messaging protocol:
        FOCUS <unit_id> <q> <r> <round>
            Establishes the squad's intended target and includes the position
            where the sender currently sees it.

        TRACK <unit_id> <q> <r> <round> <has_acted>
            Once a FOCUS target exists, any squadmate that can currently see
            that enemy reports its latest tile and whether it has already acted
            this round. Receivers keep the freshest report and can investigate
            that tile on their next activation when the target is out of sight.

        SEEN <unit_id> <q> <r> <round>
            Still accepted as a generic sighting report for compatibility.

    Messages are direct to squadmates rather than broadcasts, so the enemy does
    not receive the army's intended focus call.

    V5 keeps the V4 water-edge routing and hill-anchored archers, then adds
    soft Shield Wall footman pairs and deliberate cavalry Charge/reset cycles.
    """

    # ----------------------------------------------------------
    # Season 3 army / messaging settings
    # ----------------------------------------------------------

    MESSAGE_TTL_ROUNDS = 3
    TRACK_TTL_ROUNDS = 2
    MESSAGE_REPEAT_ROUNDS = 2
    MAX_DIRECT_MESSAGES_PER_ACTIVATION = 3
    REPORTED_FOCUS_UTILITY_BONUS = 0.22

    # Season 3 v4 strategic terrain behavior.  A long connected water component
    # is treated as a major barrier.  Footmen and cavalry are deliberately split
    # between its two endpoints so the army does not pile up against the water.
    MAJOR_WATER_MIN_TILES = 8
    WATER_GATE_RADIUS = 3

    # The Season 3 battlefield is larger than Season 2, so archers are allowed
    # to search farther for high ground.  The inherited archer behavior reads
    # these attributes dynamically, so the larger values also apply in combat.
    ARCHER_HILL_SEARCH_COST = 10.0
    ARCHER_EMERGENCY_HILL_SEARCH_COST = 12.0

    # 8 footmen must be distributed across 6 squads. Squads 0 and 1 receive
    # two footmen; every other squad receives one footman.
    FOOTMAN_TO_SQUAD = {
        0: 0,
        1: 0,
        2: 1,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5,
    }

    # ==========================================================
    # RESET / MAIN LOOP
    # ==========================================================

    def reset(self, seed, observation) -> None:
        super().reset(seed, observation)

        self.pending_messages = []
        self.reported_enemies = {}
        self.reported_focus_target_id = None
        self.reported_focus_round = -999
        self.reported_focus_tick = -1
        self.last_sent_focus_signature = None
        self.last_sent_focus_round = -999
        self.last_sent_track_signature = None
        self.last_sent_track_round = -999

        self.own_unit_id = me.unit_id(observation)
        self.own_side = me.side(observation)

        self.allied_player_ids = set()
        self.player_by_unit_id = {}
        self.unit_id_by_player = {}

        self.squad_id = self._squad_id_for_unit_id(self.own_unit_id)
        self.squad_unit_ids = set()
        self.squad_player_ids = []
        self.squad_footman_ids = set()
        self.squad_archer_ids = set()
        self.squad_cavalry_ids = set()

        self._initialize_army_structure(observation)

        # Major-water routing is precomputed once per unit instance.  Each squad
        # is assigned one of the two barrier endpoints, alternating by squad id.
        self.major_water_barrier = None
        self.water_gate_candidates = []
        self.water_barrier_crossed = False
        self._initialize_water_barrier(observation)

    def act(
        self,
        observation: SkirmishObservation,
    ) -> SkirmishAction:
        unit_type = me.unit_type(observation)
        allies = list(visible.allies(observation))
        enemies = list(visible.enemies(observation))
        squad_allies = self._visible_squad_allies(allies)
        self._update_water_crossing(observation)

        # ------------------------------------------------------
        # Visible contact always overrides a stale message.
        # ------------------------------------------------------
        if enemies:
            self.phase = "COMBAT"
            self.has_seen_enemy = True

            # If the squad already has a fresh FOCUS target and this unit can
            # see it, use the target's CURRENT visible position and report that
            # tile to the rest of the squad. This turns FOCUS into a persistent
            # squad intention and TRACK into live position updates.
            visible_focus = self._visible_focus_target(
                observation,
                enemies,
            )

            if visible_focus is not None:
                team_target = visible_focus
                self._remember_visible_target(
                    observation,
                    visible_focus,
                    kind="TRACK",
                )
                self._queue_track_messages(
                    observation,
                    visible_focus,
                )
            else:
                team_target = self._select_target(
                    observation,
                    allies,
                    enemies,
                )

                # Only establish a new squad FOCUS when the previous call is no
                # longer fresh. If the focus target is merely outside this
                # unit's vision, do not overwrite the squad plan with a local
                # opportunistic target.
                if not self._focus_is_fresh(observation):
                    self._queue_focus_messages(
                        observation,
                        team_target,
                    )

            if unit_type == "footman":
                return self._footman_tree(
                    observation,
                    allies,
                    enemies,
                    team_target,
                )

            if unit_type == "archer":
                return self._archer_tree(
                    observation,
                    allies,
                    enemies,
                    team_target,
                )

            if unit_type == "cavalry":
                return self._cavalry_tree(
                    observation,
                    allies,
                    enemies,
                    team_target,
                )

            return action.stay()

        # ------------------------------------------------------
        # No visible enemies.  Footmen and cavalry that are still on their
        # starting side of the major water barrier first commit to their squad's
        # assigned endpoint.  This prevents them from stopping along the water
        # line while trying to chase a target directly through it.
        # ------------------------------------------------------
        if (
            unit_type in ("footman", "cavalry")
            and self._needs_water_bypass(observation)
        ):
            self.combat_state = "WATER_EDGE_BYPASS"
            return self._move_via_water_edge(observation)

        # ------------------------------------------------------
        # Archers use hills as their strategic positions.  If a recent report
        # exists, they prefer a hill that can cover that reported tile instead
        # of walking off high ground directly toward the old sighting.
        # ------------------------------------------------------
        report = self._freshest_report(observation)

        if unit_type == "archer":
            report_position = (
                report["position"]
                if report is not None
                else None
            )
            hill_order = self._archer_strategic_hill_action(
                observation,
                report_position=report_position,
            )
            if hill_order is not None:
                self.combat_state = "HOLD_OR_TAKE_HILL"
                return hill_order

        # ------------------------------------------------------
        # Otherwise use a fresh squad report if one exists.  A report is only a
        # destination; invisible units cannot be named as attack targets.
        # ------------------------------------------------------
        if report is not None:
            here = me.position(observation)
            report_position = report["position"]

            if tile.distance(here, report_position) > 1:
                self.phase = "INVESTIGATE"
                self.combat_state = "INVESTIGATE_REPORT"
                return self._move_toward(
                    observation,
                    report_position,
                )

            # We reached the old sighting and still see no enemy. Clear this
            # report so the agent does not camp an empty historical tile.
            self._expire_report(report["unit_id"])

        # ------------------------------------------------------
        # Combat ended / enemy lost.
        # ------------------------------------------------------
        if self.phase in ("COMBAT", "INVESTIGATE"):
            self.phase = "RECOVER"
            self.combat_state = None

        # ------------------------------------------------------
        # Recover toward the local squad, not the whole 20-unit army.
        # ------------------------------------------------------
        if self.phase == "RECOVER":
            if squad_allies:
                nearest = self._nearest_ally_distance(
                    observation,
                    squad_allies,
                )
                if nearest <= self.RALLY_DISTANCE:
                    self.phase = "FORM_UP"
                else:
                    return self._rally(
                        observation,
                        squad_allies,
                    )
            else:
                # Do not walk the entire army toward the global center if this
                # unit temporarily loses its squad. Keep advancing in its lane.
                self.phase = "FORM_UP"

        # ------------------------------------------------------
        # Season 3 rendezvous: only local squadmates matter. If none are
        # visible, keep advancing rather than converging all 20 units on center.
        # ------------------------------------------------------
        if self.phase == "RENDEZVOUS":
            if squad_allies:
                self.phase = "FORM_UP"
            else:
                return self._advance_forward(observation)

        # ------------------------------------------------------
        # Squad-scale formation / advance.
        # ------------------------------------------------------
        if self.phase == "FORM_UP":
            if squad_allies and not self._formation_ready(
                observation,
                squad_allies,
                unit_type,
            ):
                return self._form_up(
                    observation,
                    squad_allies,
                    unit_type,
                )

            return self._advance_as_team(
                observation,
                squad_allies,
                unit_type,
            )

        return action.stay()

    # ==========================================================
    # PEER MESSAGING
    # ==========================================================

    def chat(self, inbox: list[dict]) -> list[dict]:
        """
        Read friendly peer messages, store them for the next act(), then send
        any direct squad messages queued by the action just chosen.

        FOCUS establishes the squad target. TRACK updates the last-seen tile of
        that target. Because chat() runs after act(), every received report can
        only affect the receiver's NEXT activation.
        """

        for message in inbox:
            sender = message.get("from")

            # Enemy broadcasts arrive in the same inbox. Only parse messages
            # from our own roster.
            if sender not in self.allied_player_ids:
                continue

            text = message.get("text", "")
            parts = text.split()
            message_tick = self._safe_int(message.get("tick"), -1)

            # --------------------------------------------------
            # FOCUS / SEEN <unit_id> <q> <r> <round>
            # --------------------------------------------------
            if len(parts) == 5 and parts[0] in ("FOCUS", "SEEN"):
                kind = parts[0]
                unit_id = parts[1]

                try:
                    q = int(parts[2])
                    r = int(parts[3])
                    report_round = int(parts[4])
                except ValueError:
                    continue

                self._store_enemy_report(
                    unit_id=unit_id,
                    position={"q": q, "r": r},
                    report_round=report_round,
                    kind=kind,
                    tick=message_tick,
                    has_acted=None,
                )

                if kind == "FOCUS":
                    # Prefer the freshest focus call. Tick breaks ties between
                    # multiple calls made during the same round.
                    if (report_round, message_tick) >= (
                        self.reported_focus_round,
                        self.reported_focus_tick,
                    ):
                        self.reported_focus_target_id = unit_id
                        self.reported_focus_round = report_round
                        self.reported_focus_tick = message_tick

                continue

            # --------------------------------------------------
            # TRACK <unit_id> <q> <r> <round> <has_acted>
            # --------------------------------------------------
            if len(parts) == 6 and parts[0] == "TRACK":
                unit_id = parts[1]

                try:
                    q = int(parts[2])
                    r = int(parts[3])
                    report_round = int(parts[4])
                    has_acted = bool(int(parts[5]))
                except (ValueError, TypeError):
                    continue

                # TRACK is an observation update, not a new command. Ignore a
                # track for some unrelated target when this squad already has a
                # different fresh FOCUS call.
                if (
                    self.reported_focus_target_id is not None
                    and unit_id != self.reported_focus_target_id
                    and self.reported_focus_round >= report_round - self.MESSAGE_TTL_ROUNDS
                ):
                    continue

                self._store_enemy_report(
                    unit_id=unit_id,
                    position={"q": q, "r": r},
                    report_round=report_round,
                    kind="TRACK",
                    tick=message_tick,
                    has_acted=has_acted,
                )

        outgoing = self.pending_messages[:]
        self.pending_messages = []
        return outgoing

    def _queue_focus_messages(self, observation, target):
        """Establish a squad FOCUS target and send its current tile."""

        if target is None:
            return

        current_round = self._current_round(observation)
        position = target["position"]
        signature = (
            target["unit_id"],
            position["q"],
            position["r"],
        )

        # The sender also adopts its own call immediately. Agent instances do
        # not receive their own direct message.
        self.reported_focus_target_id = target["unit_id"]
        self.reported_focus_round = current_round
        self.reported_focus_tick = -1
        self._remember_visible_target(
            observation,
            target,
            kind="FOCUS",
        )

        # Avoid sending exactly the same target/position every activation.
        if (
            signature == self.last_sent_focus_signature
            and current_round - self.last_sent_focus_round
            < self.MESSAGE_REPEAT_ROUNDS
        ):
            return

        text = (
            f"FOCUS {target['unit_id']} "
            f"{position['q']} {position['r']} {current_round}"
        )

        self._queue_direct_to_squad(text)
        self.last_sent_focus_signature = signature
        self.last_sent_focus_round = current_round

    def _queue_track_messages(self, observation, target):
        """
        Tell squadmates the focus target's CURRENT visible tile.

        Any squadmate can become the tracker. This is particularly useful when
        the original caller loses sight of the target but another squad member
        still sees it.
        """

        if target is None:
            return

        if target["unit_id"] != self.reported_focus_target_id:
            return

        current_round = self._current_round(observation)
        position = target["position"]
        has_acted = 1 if self._has_acted(target) else 0
        signature = (
            target["unit_id"],
            position["q"],
            position["r"],
            has_acted,
        )

        # Repeating an unchanged tile multiple times in the same round adds no
        # information. A changed tile or changed has_acted state is sent.
        if (
            signature == self.last_sent_track_signature
            and current_round == self.last_sent_track_round
        ):
            return

        text = (
            f"TRACK {target['unit_id']} "
            f"{position['q']} {position['r']} "
            f"{current_round} {has_acted}"
        )

        self._queue_direct_to_squad(text)
        self.last_sent_track_signature = signature
        self.last_sent_track_round = current_round

    def _queue_direct_to_squad(self, text):
        """Queue one direct protocol message for each squadmate, up to the cap."""

        own_player = self.player_by_unit_id.get(self.own_unit_id)
        recipients = sorted(
            player
            for player in self.squad_player_ids
            if player is not None and player != own_player
        )
        recipients = recipients[: self.MAX_DIRECT_MESSAGES_PER_ACTIVATION]

        self.pending_messages.extend(
            {"to": recipient, "text": text}
            for recipient in recipients
        )

    def _visible_focus_target(self, observation, enemies):
        """Return the current fresh FOCUS target if this unit can see it."""

        if not self._focus_is_fresh(observation):
            return None

        for enemy in enemies:
            if enemy["unit_id"] == self.reported_focus_target_id:
                return enemy

        return None

    def _focus_is_fresh(self, observation):
        if self.reported_focus_target_id is None:
            return False

        current_round = self._current_round(observation)
        age = current_round - self.reported_focus_round

        if age < 0 or age > self.MESSAGE_TTL_ROUNDS:
            self.reported_focus_target_id = None
            self.reported_focus_round = -999
            self.reported_focus_tick = -1
            return False

        return True

    def _remember_visible_target(self, observation, target, kind):
        self._store_enemy_report(
            unit_id=target["unit_id"],
            position={
                "q": target["position"]["q"],
                "r": target["position"]["r"],
            },
            report_round=self._current_round(observation),
            kind=kind,
            tick=-1,
            has_acted=self._has_acted(target),
        )

    def _store_enemy_report(
        self,
        unit_id,
        position,
        report_round,
        kind,
        tick,
        has_acted,
    ):
        """Store a report only when it is at least as fresh as the old one."""

        old = self.reported_enemies.get(unit_id)
        new_key = (report_round, tick)

        if old is not None:
            old_key = (old.get("round", -999), old.get("tick", -1))
            if new_key < old_key:
                return

        self.reported_enemies[unit_id] = {
            "unit_id": unit_id,
            "position": position,
            "round": report_round,
            "tick": tick,
            "kind": kind,
            "has_acted": has_acted,
        }

    def _freshest_report(self, observation):
        """
        Return the freshest usable last-seen position.

        A fresh FOCUS target is preferred. TRACK locations expire faster than
        the focus intention because an exact tile becomes stale quickly when an
        enemy moves.
        """

        current_round = self._current_round(observation)
        self._focus_is_fresh(observation)

        fresh = []
        stale_ids = []

        for unit_id, report in self.reported_enemies.items():
            age = current_round - report["round"]
            ttl = (
                self.TRACK_TTL_ROUNDS
                if report.get("kind") == "TRACK"
                else self.MESSAGE_TTL_ROUNDS
            )

            if age < 0 or age > ttl:
                stale_ids.append(unit_id)
                continue

            focus_penalty = (
                0
                if unit_id == self.reported_focus_target_id
                else 1
            )

            fresh.append(
                (
                    focus_penalty,
                    -report["round"],
                    -report.get("tick", -1),
                    unit_id,
                    report,
                )
            )

        for unit_id in stale_ids:
            self.reported_enemies.pop(unit_id, None)

        if not fresh:
            return None

        fresh.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            )
        )

        return fresh[0][4]

    def _expire_report(self, unit_id):
        self.reported_enemies.pop(unit_id, None)

        # Reaching an empty old sighting clears the focus too. This prevents the
        # squad from repeatedly returning to a tile where the target has gone.
        if self.reported_focus_target_id == unit_id:
            self.reported_focus_target_id = None
            self.reported_focus_round = -999
            self.reported_focus_tick = -1

    def _safe_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    # ==========================================================
    # 20-UNIT SQUAD STRUCTURE
    # ==========================================================

    def _initialize_army_structure(self, observation):
        raw = observation.get(
            "observation",
            observation,
        )

        rosters = raw.get("rosters", {})
        own_roster = rosters.get(self.own_side, ())

        for entry in own_roster:
            unit_id = entry.get("unit_id")
            player = entry.get("player")

            if unit_id is None or player is None:
                continue

            self.allied_player_ids.add(player)
            self.player_by_unit_id[unit_id] = player
            self.unit_id_by_player[player] = unit_id

            squad_id = self._squad_id_for_unit_id(unit_id)
            if squad_id != self.squad_id:
                continue

            self.squad_unit_ids.add(unit_id)
            self.squad_player_ids.append(player)

            unit_type, _index = self._unit_type_and_index(unit_id)

            if unit_type == "footman":
                self.squad_footman_ids.add(unit_id)
            elif unit_type == "archer":
                self.squad_archer_ids.add(unit_id)
            elif unit_type == "cavalry":
                self.squad_cavalry_ids.add(unit_id)

    def _unit_type_and_index(self, unit_id):
        parts = unit_id.split("_")

        if len(parts) < 3:
            return None, None

        unit_type = parts[-2]

        try:
            index = int(parts[-1])
        except ValueError:
            return unit_type, None

        return unit_type, index

    def _squad_id_for_unit_id(self, unit_id):
        unit_type, index = self._unit_type_and_index(unit_id)

        if index is None:
            return 0

        if unit_type == "footman":
            return self.FOOTMAN_TO_SQUAD.get(index, index % 6)

        if unit_type in ("archer", "cavalry"):
            return index % 6

        return index % 6

    def _visible_squad_allies(self, allies):
        if not self.squad_unit_ids:
            return list(allies)

        return [
            ally
            for ally in allies
            if ally["unit_id"] in self.squad_unit_ids
        ]

    def _find_footman(self, allies):
        """
        Prefer this agent's designated squad footman instead of whichever of the
        eight army footmen happens to appear first in visible.allies().
        """

        squad_footmen = [
            ally
            for ally in allies
            if (
                ally["type"] == "footman"
                and ally["unit_id"] in self.squad_footman_ids
            )
        ]

        here = None
        if squad_footmen:
            # In squads 0 and 1, prefer the lowest-index footman as the stable
            # formation anchor. The second footman remains free to advance.
            return min(
                squad_footmen,
                key=lambda ally: (
                    self._unit_type_and_index(ally["unit_id"])[1],
                    ally["unit_id"],
                ),
            )

        # If the designated footman is dead or outside vision, using the nearest
        # visible footman is better than freezing the archer/cavalry.
        visible_footmen = [
            ally
            for ally in allies
            if ally["type"] == "footman"
        ]

        if not visible_footmen:
            return None

        # Without observation here, stable unit id is a deterministic fallback.
        return min(
            visible_footmen,
            key=lambda ally: ally["unit_id"],
        )

    def _cavalry_threat_pair(self, allies, enemies):
        """
        Season 3 army adaptation: a frontline unit protects its own squad archer
        first instead of abandoning formation for any of the six visible archers.
        """

        archers = [
            ally
            for ally in allies
            if (
                ally["type"] == "archer"
                and (
                    not self.squad_archer_ids
                    or ally["unit_id"] in self.squad_archer_ids
                )
            )
        ]

        if not archers:
            return None

        candidates = []

        for enemy in enemies:
            if enemy["type"] != "cavalry":
                continue

            for archer in archers:
                distance = tile.distance(
                    enemy["position"],
                    archer["position"],
                )

                if distance <= self.ARCHER_CAVALRY_THREAT:
                    candidates.append(
                        (
                            distance,
                            enemy["hit_points"],
                            enemy["unit_id"],
                            enemy,
                            archer,
                        )
                    )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )

        best = candidates[0]
        return best[3], best[4]

    # ==========================================================
    # SQUAD FORMATION / ADVANCE
    # ==========================================================

    def _rendezvous(self, observation):
        # Global-center rendezvous is unsafe with 20 units. If a squad footman
        # is visible, move toward it. Otherwise keep progressing down the unit's
        # own lane toward enemy territory.
        allies = list(visible.allies(observation))
        squad_allies = self._visible_squad_allies(allies)
        footman = self._find_footman(squad_allies)

        if footman is not None:
            return self._move_toward(
                observation,
                footman["position"],
            )

        return self._advance_forward(observation)

    def _rally(self, observation, allies):
        if not allies:
            return self._advance_forward(observation)

        footman = self._find_footman(allies)

        if footman is not None:
            goal = footman["position"]
        else:
            here = me.position(observation)
            nearest = min(
                allies,
                key=lambda ally: tile.distance(
                    here,
                    ally["position"],
                ),
            )
            goal = nearest["position"]

        return self._move_toward(
            observation,
            goal,
        )

    def _formation_ready(
        self,
        observation,
        allies,
        unit_type,
    ):
        # Footmen are the squad anchors. Shield-wall spacing is intentionally
        # deferred to the next Season 3 version.
        if unit_type == "footman":
            return True

        footman = self._find_footman(allies)

        # Do not freeze if the squad anchor is temporarily outside vision.
        if footman is None:
            return True

        here = me.position(observation)

        if unit_type == "archer":
            desired = self._behind_position(
                observation,
                footman["position"],
            )
            return tile.distance(here, desired) <= 1

        if unit_type == "cavalry":
            desired = self._cavalry_formation_position(
                observation,
                footman["position"],
            )
            return tile.distance(here, desired) <= 1

        return True

    def _form_up(
        self,
        observation,
        allies,
        unit_type,
    ):
        if unit_type == "footman":
            return self._advance_forward(observation)

        footman = self._find_footman(allies)

        if footman is None:
            return self._advance_forward(observation)

        if unit_type == "archer":
            goal = self._behind_position(
                observation,
                footman["position"],
            )
            return self._move_toward(
                observation,
                goal,
            )

        if unit_type == "cavalry":
            goal = self._cavalry_formation_position(
                observation,
                footman["position"],
            )
            return self._move_toward(
                observation,
                goal,
            )

        return self._advance_forward(observation)

    def _advance_as_team(
        self,
        observation,
        allies,
        unit_type,
    ):
        if unit_type == "footman":
            return self._advance_forward(observation)

        footman = self._find_footman(allies)

        if footman is None:
            return self._advance_forward(observation)

        if unit_type == "archer":
            goal = self._behind_position(
                observation,
                footman["position"],
            )
            return self._move_toward(
                observation,
                goal,
            )

        if unit_type == "cavalry":
            goal = self._cavalry_formation_position(
                observation,
                footman["position"],
            )
            return self._move_toward(
                observation,
                goal,
            )

        return self._advance_forward(observation)

    # ==========================================================
    # MESSAGE-AWARE UTILITY
    # ==========================================================

    def _score_enemy_utility(
        self,
        observation,
        allies,
        enemy,
    ):
        score = super()._score_enemy_utility(
            observation,
            allies,
            enemy,
        )

        current_round = self._current_round(observation)
        focus_is_fresh = (
            self.reported_focus_target_id is not None
            and current_round - self.reported_focus_round
            <= self.MESSAGE_TTL_ROUNDS
        )

        if (
            focus_is_fresh
            and enemy["unit_id"] == self.reported_focus_target_id
        ):
            score += self.REPORTED_FOCUS_UTILITY_BONUS

        return score

    def _current_round(self, observation):
        raw = observation.get(
            "observation",
            observation,
        )
        return int(raw.get("round", 0))


    # ==========================================================
    # SEASON 3 V3: ACTIVE-SQUAD OVERRIDES
    # ==========================================================
    #
    # These methods deliberately override the inherited Season 2-style
    # formation/combat rules above.  With 20 units, exact formation tiles and
    # "wait for the footman" behavior create traffic jams.  V3 therefore uses
    # loose squad spacing and makes melee units fight immediately when contact
    # occurs.  The FOCUS/TRACK messaging system from V2 is left unchanged.

    LOOSE_ARCHER_DISTANCE = 3
    LOOSE_CAVALRY_DISTANCE = 4

    # Season 3 abilities. Shield Wall is a soft preference: footmen still take
    # immediate attacks and kills instead of refusing combat to preserve a pair.
    SHIELD_WALL_ASSIGNED_BONUS = 0.42
    SHIELD_WALL_ANY_FOOTMAN_BONUS = 0.16

    # Cavalry deals 3 normally and 5 when Charge applies. The rules award Charge
    # when the cavalry ENDS at least 3 hexes from its activation start; this is
    # displacement, not merely the number of encoded path steps. Forested targets
    # do not receive charge-bonus damage.
    CAVALRY_CHARGE_BONUS_DAMAGE = 2
    CAVALRY_CHARGE_MIN_DISPLACEMENT = 3
    CAVALRY_RESET_IDEAL_DISTANCE = 4

    def _find_footman(self, allies):
        """
        Return only a footman assigned to THIS unit's squad.

        V2 fell back to any visible army footman when the designated squad
        footman was out of sight.  In a 20-unit army that caused neighboring
        squads to merge into one large cluster.  V3 keeps squads independent:
        if the squad footman is not visible, return None and let the unit keep
        moving instead of attaching itself to another squad.
        """
        squad_footmen = [
            ally
            for ally in allies
            if (
                ally["type"] == "footman"
                and ally["unit_id"] in self.squad_footman_ids
            )
        ]

        if not squad_footmen:
            return None

        return min(
            squad_footmen,
            key=lambda ally: (
                self._unit_type_and_index(ally["unit_id"])[1],
                ally["unit_id"],
            ),
        )

    def _footman_tree(
        self,
        observation,
        allies,
        enemies,
        team_target,
    ):
        """
        Season 3 v5 footman behavior with a soft Shield Wall preference.

        Immediate combat still wins over formation:
            1. attack an adjacent enemy;
            2. take a guaranteed reachable kill;
            3. cross the assigned water edge when still on the home side;
            4. intercept an urgent cavalry threat to the squad archer;
            5. otherwise advance/attack while preferring endpoints adjacent to
               the assigned footman partner.

        This exploits Shield Wall without recreating the rigid formation jams
        from the first 20-unit implementation.
        """
        here = me.position(observation)

        adjacent = [
            enemy
            for enemy in enemies
            if tile.distance(here, enemy["position"]) <= 1
        ]

        if adjacent:
            enemy = min(
                adjacent,
                key=lambda item: (
                    0 if item["hit_points"] <= self.FOOTMAN_DAMAGE else 1,
                    0 if item["unit_id"] == self.reported_focus_target_id else 1,
                    item["hit_points"],
                    0 if item["type"] == "cavalry" else 1,
                    item["unit_id"],
                ),
            )
            self.combat_state = "MELEE_SHIELD_WALL"
            return action.stay(
                enemy["unit_id"],
                observation,
            )

        kill_order = self._footman_reachable_kill_order(
            observation,
            allies,
            enemies,
        )

        if kill_order is not None:
            enemy, path_id = kill_order
            self.combat_state = "FINISH_SHIELD_WALL"

            if path_id == 0:
                return action.stay(
                    enemy["unit_id"],
                    observation,
                )

            return action.move(
                path_id,
                enemy["unit_id"],
                observation,
            )

        if self._needs_water_bypass(observation):
            self.combat_state = "WATER_EDGE_BYPASS"
            return self._move_via_water_edge(
                observation,
                target_id=team_target["unit_id"],
            )

        threat_pair = self._urgent_cavalry_threat_pair(
            observation,
            allies,
            enemies,
        )

        if threat_pair is not None:
            threat, archer = threat_pair
            self.combat_state = "SCREEN"
            return self._footman_screen_archer(
                observation,
                threat,
                archer,
            )

        self.combat_state = "ADVANCE_SHIELD_WALL"
        return self._footman_active_attack(
            observation,
            allies,
            enemies,
            team_target,
        )

    def _cavalry_tree(
        self,
        observation,
        allies,
        enemies,
        team_target,
    ):
        """
        Season 3 v5 cavalry behavior that deliberately exploits Charge.

        Charge is awarded when the cavalry ends at least three hexes from its
        activation start and then attacks a non-forest target. A successful
        charge raises cavalry damage from 3 to 5. After a charge the cavalry
        normally disengages to create enough separation for another charge.
        """
        here = me.position(observation)
        adjacent = [
            enemy
            for enemy in enemies
            if tile.distance(here, enemy["position"]) <= 1
        ]

        # Do not abandon a guaranteed immediate kill just because the previous
        # activation was a charge.
        killable_adjacent = [
            enemy
            for enemy in adjacent
            if enemy["hit_points"] <= self.CAVALRY_DAMAGE
        ]
        if killable_adjacent:
            enemy = min(
                killable_adjacent,
                key=lambda item: (
                    0 if item["unit_id"] == self.reported_focus_target_id else 1,
                    item["hit_points"],
                    item["unit_id"],
                ),
            )
            self.combat_state = "MELEE_FINISH"
            return action.stay(
                enemy["unit_id"],
                observation,
            )

        # A charge-reset cycle prevents cavalry from turning into a stationary
        # footman after its first impact. If a safe reset is available, create
        # 3-5 tiles of separation and prepare another charge.
        if self.combat_state == "CHARGE":
            reset_order = self._cavalry_charge_reset(
                observation,
                enemies,
                preferred_target=(
                    min(adjacent, key=lambda e: (e["hit_points"], e["unit_id"]))
                    if adjacent
                    else team_target
                ),
            )
            if reset_order is not None:
                self.combat_state = "RESET_CHARGE"
                return reset_order

        kill_order = self._cavalry_reachable_kill_order(
            observation,
            enemies,
        )

        if kill_order is not None:
            enemy, path_id, is_charge = kill_order
            self.combat_state = "CHARGE" if is_charge else "FINISH"

            if path_id == 0:
                return action.stay(
                    enemy["unit_id"],
                    observation,
                )

            return action.move(
                path_id,
                enemy["unit_id"],
                observation,
            )

        if self._needs_water_bypass(observation):
            self.combat_state = "WATER_EDGE_BYPASS"
            return self._move_via_water_edge(
                observation,
                target_id=team_target["unit_id"],
            )

        threat_pair = self._urgent_cavalry_threat_pair(
            observation,
            allies,
            enemies,
        )

        if threat_pair is not None:
            threat, _archer = threat_pair
            self.combat_state = "COUNTER_CAVALRY"
            return self._cavalry_active_attack(
                observation,
                threat,
                enemies,
            )

        # If cavalry is stuck in melee with a healthy enemy, prefer a safe reset
        # over repeatedly making 3-damage stationary attacks. If no useful reset
        # exists, keep fighting rather than freeze.
        if adjacent:
            enemy = min(
                adjacent,
                key=lambda item: (
                    0 if item["unit_id"] == self.reported_focus_target_id else 1,
                    item["hit_points"],
                    item["unit_id"],
                ),
            )
            reset_order = self._cavalry_charge_reset(
                observation,
                enemies,
                preferred_target=enemy,
            )
            if reset_order is not None:
                self.combat_state = "SETUP_CHARGE"
                return reset_order

            self.combat_state = "MELEE"
            return action.stay(
                enemy["unit_id"],
                observation,
            )

        self.combat_state = "ATTACK"
        return self._cavalry_active_attack(
            observation,
            team_target,
            enemies,
        )

    def _urgent_cavalry_threat_pair(
        self,
        observation,
        allies,
        enemies,
    ):
        """
        Return a cavalry threat only when it is close enough to justify pulling
        a melee unit away from its current attack.

        * unacted cavalry within 4 tiles of the squad archer is urgent because
          it can still use its movement this round;
        * cavalry that already acted is urgent only within 2 tiles.
        """
        archers = [
            ally
            for ally in allies
            if (
                ally["type"] == "archer"
                and ally["unit_id"] in self.squad_archer_ids
            )
        ]

        if not archers:
            return None

        candidates = []

        for enemy in enemies:
            if enemy["type"] != "cavalry":
                continue

            for archer in archers:
                distance = tile.distance(
                    enemy["position"],
                    archer["position"],
                )

                has_acted = self._has_acted(enemy)
                urgent_limit = 2 if has_acted else 4

                if distance > urgent_limit:
                    continue

                candidates.append(
                    (
                        1 if has_acted else 0,
                        distance,
                        enemy["hit_points"],
                        enemy["unit_id"],
                        enemy,
                        archer,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
                item[3],
            )
        )

        best = candidates[0]
        return best[4], best[5]

    def _cavalry_active_attack(
        self,
        observation,
        target,
        enemies,
    ):
        """
        Score every legal cavalry path with a strong preference for a real
        Charge attack: end adjacent to the target after >=3 hexes of displacement.

        Short melee attacks remain legal and useful, but an equally safe Charge
        is preferred because Season 3 raises the strike from 3 to 5 damage.
        """
        here = me.position(observation)
        route_costs = self._route_cost_map(
            observation,
            target["position"],
        )
        best_path = 0
        best_score = float("-inf")
        best_is_charge = False

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(
                here,
                path_id,
            )
            target_distance = tile.distance(
                destination,
                target["position"],
            )
            displacement = tile.distance(
                here,
                destination,
            )
            step_count = len(paths.decode(path_id))

            route_cost = route_costs.get(
                self._position_key(destination),
                float("inf"),
            )
            if route_cost == float("inf"):
                continue

            route_utility = 1.0 / (1.0 + route_cost)

            adjacent_enemies = sum(
                1
                for enemy in enemies
                if tile.distance(destination, enemy["position"]) <= 1
            )
            safety_utility = 1.0 - min(
                1.0,
                adjacent_enemies / 3.0,
            )

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="cavalry",
                purpose="mobility",
            )

            in_melee = target_distance <= 1
            is_charge = self._cavalry_path_is_charge(
                observation,
                here,
                destination,
                target,
            )

            # A distance of roughly 3-4 from the target is a useful reset/setup
            # position when this activation cannot actually connect a charge.
            setup_utility = 1.0 - min(
                1.0,
                abs(target_distance - self.CAVALRY_RESET_IDEAL_DISTANCE) / 4.0,
            )
            if in_melee:
                setup_utility = 0.0

            movement_utility = min(1.0, displacement / 4.0)

            score = (
                0.24 * route_utility
                + 0.18 * safety_utility
                + 0.08 * terrain_utility
                + 0.08 * movement_utility
                + 0.12 * setup_utility
            )

            if in_melee:
                score += 0.55

            if is_charge:
                score += 0.90

            # Standing still while out of melee wastes cavalry mobility.
            if path_id == 0 and not in_melee:
                score -= 0.50

            # A charge into a crowd is usually not worth one stronger hit.
            if adjacent_enemies >= 2:
                score -= 0.75 * (adjacent_enemies - 1)

            # Tiny tie-break preference for using more of the available move.
            score += 0.01 * step_count

            if score > best_score:
                best_score = score
                best_path = path_id
                best_is_charge = is_charge

        if best_path == 0:
            self.combat_state = "MELEE" if tile.distance(here, target["position"]) <= 1 else "ATTACK"
            return action.stay(
                target["unit_id"],
                observation,
            )

        self.combat_state = "CHARGE" if best_is_charge else "ATTACK"
        return action.move(
            best_path,
            target["unit_id"],
            observation,
        )
    # ==========================================================
    # SEASON 3 V5: SHIELD WALL + CAVALRY CHARGE HELPERS
    # ==========================================================

    def _footman_partner_id(self):
        """Return this footman's deterministic Shield Wall partner unit id."""
        unit_type, index = self._unit_type_and_index(self.own_unit_id)
        if unit_type != "footman" or index is None:
            return None

        partner_index = index + 1 if index % 2 == 0 else index - 1
        return f"{self.own_side}_footman_{partner_index}"

    def _visible_footman_partner(self, allies):
        partner_id = self._footman_partner_id()
        if partner_id is None:
            return None

        for ally in allies:
            if ally["unit_id"] == partner_id:
                return ally
        return None

    def _shield_wall_value(self, destination, allies):
        """
        Return a soft tactical bonus for ending beside a footman.

        The assigned pair receives the large bonus. Any other allied footman
        still creates the actual Shield Wall rule, so it receives a smaller
        bonus rather than being ignored completely.
        """
        partner_id = self._footman_partner_id()
        assigned_adjacent = False
        any_footman_adjacent = False

        for ally in allies:
            if ally.get("type") != "footman":
                continue
            if tile.distance(destination, ally["position"]) != 1:
                continue

            any_footman_adjacent = True
            if ally["unit_id"] == partner_id:
                assigned_adjacent = True

        if assigned_adjacent:
            return self.SHIELD_WALL_ASSIGNED_BONUS
        if any_footman_adjacent:
            return self.SHIELD_WALL_ANY_FOOTMAN_BONUS
        return 0.0

    def _footman_reachable_kill_order(
        self,
        observation,
        allies,
        enemies,
    ):
        """Take a guaranteed kill; among kill paths prefer Shield Wall safety."""
        here = me.position(observation)
        legal_paths = list(action.legal_paths(observation))
        best = None
        best_key = None

        for enemy in enemies:
            if enemy["hit_points"] > self.FOOTMAN_DAMAGE:
                continue

            for path_id in legal_paths:
                destination = tile.at_path_end(here, path_id)
                if tile.distance(destination, enemy["position"]) > self.FOOTMAN_RANGE:
                    continue

                shield_value = self._shield_wall_value(destination, allies)
                adjacent_enemies = sum(
                    1
                    for other in enemies
                    if tile.distance(destination, other["position"]) <= 1
                )
                step_count = len(paths.decode(path_id))

                key = (
                    enemy["hit_points"],
                    -shield_value,
                    adjacent_enemies,
                    step_count,
                    enemy["unit_id"],
                    path_id,
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best = (enemy, path_id)

        return best

    def _footman_active_attack(
        self,
        observation,
        allies,
        enemies,
        target,
    ):
        """
        Advance on a visible target while treating Shield Wall as a utility
        bonus rather than a mandatory formation rule.
        """
        here = me.position(observation)
        route_costs = self._route_cost_map(observation, target["position"])
        best_path = 0
        best_score = float("-inf")

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(here, path_id)
            route_cost = route_costs.get(
                self._position_key(destination),
                float("inf"),
            )
            if route_cost == float("inf"):
                continue

            target_distance = tile.distance(destination, target["position"])
            route_utility = 1.0 / (1.0 + route_cost)
            shield_utility = self._shield_wall_value(destination, allies)

            adjacent_enemies = sum(
                1
                for enemy in enemies
                if tile.distance(destination, enemy["position"]) <= 1
            )
            safety_utility = 1.0 - min(1.0, adjacent_enemies / 3.0)

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="footman",
                purpose="attack",
            )
            step_count = len(paths.decode(path_id))

            score = (
                0.48 * route_utility
                + shield_utility
                + 0.17 * safety_utility
                + 0.10 * terrain_utility
                + 0.03 * min(1.0, step_count / 2.0)
            )

            if target_distance <= 1:
                score += 0.65

            if path_id == 0 and target_distance > 1:
                score -= 0.35

            if score > best_score:
                best_score = score
                best_path = path_id

        if best_path == 0:
            return action.stay(target["unit_id"], observation)

        return action.move(
            best_path,
            target["unit_id"],
            observation,
        )

    def _footman_advance_with_shield_wall(self, observation, allies):
        """Forward strategic movement with a soft preference for the paired footman."""
        here = me.position(observation)
        strategic_goal = tile.at_mirror(here, observation)
        route_costs = self._route_cost_map(observation, strategic_goal)
        all_allies = list(visible.allies(observation))

        best_path = 0
        best_score = float("inf")

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(here, path_id)
            route_cost = route_costs.get(
                self._position_key(destination),
                float("inf"),
            )
            if route_cost == float("inf"):
                continue

            shield_value = self._shield_wall_value(destination, all_allies)
            congestion = sum(
                1
                for ally in all_allies
                if tile.distance(destination, ally["position"]) <= 1
            )
            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="footman",
                purpose="mobility",
            )
            step_count = len(paths.decode(path_id))

            score = (
                route_cost
                + 0.30 * congestion
                - 1.20 * shield_value
                - 0.25 * terrain_utility
                - 0.06 * step_count
            )
            if path_id == 0:
                score += 0.25

            if score < best_score:
                best_score = score
                best_path = path_id

        if best_path == 0:
            return action.stay()
        return action.move(best_path)

    def _target_blocks_charge_bonus(self, observation, target):
        info = tile.terrain_at(observation, target["position"])
        return bool(info and info.get("feature") == "forest")

    def _cavalry_path_is_charge(
        self,
        observation,
        start,
        destination,
        target,
    ):
        """True only when the path produces the actual Season 3 Charge bonus."""
        if tile.distance(destination, target["position"]) > self.CAVALRY_RANGE:
            return False
        if tile.distance(start, destination) < self.CAVALRY_CHARGE_MIN_DISPLACEMENT:
            return False
        if self._target_blocks_charge_bonus(observation, target):
            return False
        return True

    def _cavalry_attack_damage_for_path(
        self,
        observation,
        start,
        destination,
        target,
    ):
        damage = self.CAVALRY_DAMAGE
        if self._cavalry_path_is_charge(
            observation,
            start,
            destination,
            target,
        ):
            damage += self.CAVALRY_CHARGE_BONUS_DAMAGE
        return damage

    def _cavalry_reachable_kill_order(self, observation, enemies):
        """Find kills using 3 damage normally or 5 damage for a valid Charge."""
        here = me.position(observation)
        best = None
        best_key = None

        for enemy in enemies:
            for path_id in action.legal_paths(observation):
                destination = tile.at_path_end(here, path_id)
                if tile.distance(destination, enemy["position"]) > self.CAVALRY_RANGE:
                    continue

                damage = self._cavalry_attack_damage_for_path(
                    observation,
                    here,
                    destination,
                    enemy,
                )
                if enemy["hit_points"] > damage:
                    continue

                is_charge = damage > self.CAVALRY_DAMAGE
                adjacent_enemies = sum(
                    1
                    for other in enemies
                    if tile.distance(destination, other["position"]) <= 1
                )
                displacement = tile.distance(here, destination)

                key = (
                    1 if adjacent_enemies >= 2 else 0,
                    0 if is_charge else 1,
                    enemy["hit_points"],
                    adjacent_enemies,
                    -displacement,
                    enemy["unit_id"],
                    path_id,
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best = (enemy, path_id, is_charge)

        return best

    def _cavalry_charge_reset(
        self,
        observation,
        enemies,
        preferred_target,
    ):
        """
        Break contact and create roughly 3-5 tiles of separation for the next
        charge. Return None if no move is materially safer than staying.
        """
        here = me.position(observation)
        best_path = None
        best_score = float("-inf")

        for path_id in action.legal_paths(observation):
            if path_id == 0:
                continue

            destination = tile.at_path_end(here, path_id)
            nearest_enemy = min(
                tile.distance(destination, enemy["position"])
                for enemy in enemies
            )
            target_distance = tile.distance(
                destination,
                preferred_target["position"],
            )

            # Do not call a move a reset if it remains in melee.
            if nearest_enemy <= 1:
                continue

            reset_utility = 1.0 - min(
                1.0,
                abs(target_distance - self.CAVALRY_RESET_IDEAL_DISTANCE) / 4.0,
            )
            safety_utility = min(1.0, nearest_enemy / 5.0)
            displacement = tile.distance(here, destination)
            movement_utility = min(1.0, displacement / 4.0)

            info = tile.terrain_at(observation, destination) or {}
            open_ground_bonus = 0.12 if info.get("feature") == "none" else 0.0
            marsh_penalty = 0.25 if info.get("feature") == "marsh" else 0.0

            score = (
                0.45 * reset_utility
                + 0.35 * safety_utility
                + 0.08 * movement_utility
                + open_ground_bonus
                - marsh_penalty
            )

            if score > best_score:
                best_score = score
                best_path = path_id

        if best_path is None:
            return None
        return action.move(best_path)


    def _formation_ready(
        self,
        observation,
        allies,
        unit_type,
    ):
        """Season 3 v5 formation test with hill-anchored archers and paired footmen."""
        if unit_type == "footman":
            return True

        here = me.position(observation)

        # An archer is considered strategically formed only when it has high
        # ground.  This makes hill occupation more important than standing a
        # fixed distance behind a footman.
        if unit_type == "archer":
            return self._is_hill(observation, here)

        footman = self._find_footman(allies)
        if footman is None:
            return True

        if unit_type == "cavalry":
            return tile.distance(
                here,
                footman["position"],
            ) <= self.LOOSE_CAVALRY_DISTANCE

        return True

    def _form_up(
        self,
        observation,
        allies,
        unit_type,
    ):
        """Form loosely, preserving water routing and soft Shield Wall pairs."""
        if unit_type in ("footman", "cavalry") and self._needs_water_bypass(observation):
            return self._move_via_water_edge(observation)

        if unit_type == "archer":
            hill_order = self._archer_strategic_hill_action(observation)
            if hill_order is not None:
                return hill_order

        if unit_type == "footman":
            return self._footman_advance_with_shield_wall(
                observation,
                allies,
            )

        footman = self._find_footman(allies)
        if footman is None:
            return self._advance_forward(observation)

        return self._move_near_position(
            observation,
            footman["position"],
            desired_distance=3,
            unit_type=unit_type,
        )

    def _advance_as_team(
        self,
        observation,
        allies,
        unit_type,
    ):
        """Advance by role: paired footmen, charging cavalry, hill archers."""
        if unit_type in ("footman", "cavalry") and self._needs_water_bypass(observation):
            return self._move_via_water_edge(observation)

        if unit_type == "archer":
            hill_order = self._archer_strategic_hill_action(observation)
            if hill_order is not None:
                return hill_order

        if unit_type == "footman":
            return self._footman_advance_with_shield_wall(
                observation,
                allies,
            )

        footman = self._find_footman(allies)
        return self._loose_forward_move(
            observation,
            footman,
            unit_type,
        )

    def _rally(self, observation, allies):
        """Recover by role without undoing water routing or Shield Wall pairing."""
        unit_type = me.unit_type(observation)

        if unit_type in ("footman", "cavalry") and self._needs_water_bypass(observation):
            return self._move_via_water_edge(observation)

        if unit_type == "archer":
            hill_order = self._archer_strategic_hill_action(observation)
            if hill_order is not None:
                return hill_order

        if not allies:
            return self._advance_forward(observation)

        if unit_type == "footman":
            return self._footman_advance_with_shield_wall(
                observation,
                allies,
            )

        footman = self._find_footman(allies)
        if footman is None:
            return self._advance_forward(observation)

        return self._move_near_position(
            observation,
            footman["position"],
            desired_distance=3,
            unit_type=unit_type,
        )

    def _move_near_position(
        self,
        observation,
        anchor_position,
        desired_distance,
        unit_type,
    ):
        """
        Choose a legal endpoint near an anchor without trying to occupy the
        anchor itself.  This is congestion-resistant on a crowded army board.
        """
        here = me.position(observation)
        all_allies = list(visible.allies(observation))

        best_path = 0
        best_key = None

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(
                here,
                path_id,
            )

            anchor_distance = tile.distance(
                destination,
                anchor_position,
            )

            spacing_error = abs(
                anchor_distance - desired_distance
            )

            nearby_friendlies = sum(
                1
                for ally in all_allies
                if tile.distance(
                    destination,
                    ally["position"],
                ) <= 1
            )

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type=unit_type,
                purpose="mobility",
            )

            step_count = len(paths.decode(path_id))

            key = (
                spacing_error,
                nearby_friendlies,
                -terrain_utility,
                -step_count,
                path_id,
            )

            if best_key is None or key < best_key:
                best_key = key
                best_path = path_id

        if best_path == 0:
            return action.stay()

        return action.move(best_path)

    def _loose_forward_move(
        self,
        observation,
        footman,
        unit_type,
    ):
        """
        Water-aware forward movement with soft squad spacing and anti-congestion.

        The strategic destination remains enemy territory, so a squad can route
        around water.  Footman distance is only a penalty, not a hard leash.
        """
        here = me.position(observation)
        strategic_goal = tile.at_mirror(
            here,
            observation,
        )
        route_costs = self._route_cost_map(
            observation,
            strategic_goal,
        )
        all_allies = list(visible.allies(observation))

        desired_distance = (
            2.0
            if unit_type == "archer"
            else 3.0
        )

        forward = me.direction(observation)
        best_path = 0
        best_score = float("inf")

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(
                here,
                path_id,
            )

            route_cost = route_costs.get(
                self._position_key(destination),
                float("inf"),
            )

            if route_cost == float("inf"):
                continue

            support_penalty = 0.0
            ahead_penalty = 0.0

            if footman is not None:
                support_distance = tile.distance(
                    destination,
                    footman["position"],
                )
                support_penalty = 0.65 * abs(
                    support_distance - desired_distance
                )

                # Archers should generally remain behind their squad footman,
                # but this is a soft preference rather than an exact tile.
                if unit_type == "archer":
                    if forward == 2:
                        ahead = (
                            destination["q"]
                            - footman["position"]["q"]
                        )
                    else:
                        ahead = (
                            footman["position"]["q"]
                            - destination["q"]
                        )

                    if ahead > 1:
                        ahead_penalty = 1.25 * (ahead - 1)

            nearby_friendlies = sum(
                1
                for ally in all_allies
                if tile.distance(
                    destination,
                    ally["position"],
                ) <= 1
            )

            congestion_penalty = 0.55 * nearby_friendlies

            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type=unit_type,
                purpose="mobility",
            )

            step_count = len(paths.decode(path_id))

            score = (
                route_cost
                + support_penalty
                + ahead_penalty
                + congestion_penalty
                - 0.35 * terrain_utility
                - 0.08 * step_count
            )

            # If moving and staying are otherwise similar, prefer movement so
            # the army does not form a stationary traffic jam.
            if path_id == 0:
                score += 0.30

            if score < best_score:
                best_score = score
                best_path = path_id

        if best_path == 0:
            return action.stay()

        return action.move(best_path)


    # ==========================================================
    # SEASON 3 V5: MAJOR WATER-BARRIER ROUTING
    # ==========================================================

    def _initialize_water_barrier(self, observation):
        """
        Detect the largest connected water component and treat it as the main
        barrier when it is long enough.  Squads alternate between its two ends.

        The Season 3 sides advance east/west, so a long north/south water line
        is crossed when q moves from the home side to the enemy side.  The
        endpoint search itself is geometry-based and does not assume a fixed r.
        """
        raw = observation.get("observation", observation)
        battlefield = raw.get("battlefield", {})
        grid = battlefield.get("tiles", [])

        water = set()
        for r, row in enumerate(grid):
            for q, info in enumerate(row):
                if isinstance(info, dict) and info.get("terrain") == "water":
                    water.add((q, r))

        if not water:
            return

        components = []
        remaining = set(water)

        while remaining:
            start = remaining.pop()
            component = {start}
            frontier = [start]

            while frontier:
                q, r = frontier.pop()
                position = {"q": q, "r": r}
                for neighbor in tile.neighbors(position).values():
                    key = (neighbor["q"], neighbor["r"])
                    if key in remaining:
                        remaining.remove(key)
                        component.add(key)
                        frontier.append(key)

            components.append(component)

        component = max(components, key=len)
        if len(component) < self.MAJOR_WATER_MIN_TILES:
            return

        positions = [
            {"q": q, "r": r}
            for q, r in component
        ]

        # Prefer true line endpoints (few water neighbors), then take the most
        # widely separated pair.  This also works for a slightly bent barrier.
        endpoint_pool = []
        for position in positions:
            degree = sum(
                1
                for neighbor in tile.neighbors(position).values()
                if (neighbor["q"], neighbor["r"]) in component
            )
            if degree <= 2:
                endpoint_pool.append(position)

        if len(endpoint_pool) < 2:
            endpoint_pool = positions

        best_pair = None
        best_distance = -1
        for i, first in enumerate(endpoint_pool):
            for second in endpoint_pool[i + 1:]:
                distance = tile.distance(first, second)
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (first, second)

        if best_pair is None:
            return

        endpoints = sorted(
            best_pair,
            key=lambda pos: (pos["r"], pos["q"]),
        )

        barrier_q = sum(q for q, _r in component) / float(len(component))
        forward_sign = 1 if me.direction(observation) == 2 else -1

        gate_sets = []
        for endpoint in endpoints:
            gates = self._far_side_gate_candidates(
                observation,
                endpoint,
                barrier_q,
                forward_sign,
            )
            gate_sets.append(gates)

        if not any(gate_sets):
            return

        self.major_water_barrier = {
            "tiles": component,
            "endpoints": endpoints,
            "barrier_q": barrier_q,
            "forward_sign": forward_sign,
        }

        # Cavalry keeps the alternating squad split. Footmen are assigned by
        # Shield Wall PAIR so partners 0/1, 2/3, 4/5, and 6/7 always use the same
        # water edge instead of being separated by the barrier-routing policy.
        own_type, own_index = self._unit_type_and_index(self.own_unit_id)
        if own_type == "footman" and own_index is not None:
            preferred_edge = (own_index // 2) % 2
        else:
            preferred_edge = self.squad_id % 2
        gates = gate_sets[preferred_edge]
        if not gates:
            gates = gate_sets[1 - preferred_edge]

        self.water_gate_candidates = gates
        self._update_water_crossing(observation)

    def _far_side_gate_candidates(
        self,
        observation,
        endpoint,
        barrier_q,
        forward_sign,
    ):
        """Return several passable enemy-side tiles near one water endpoint."""
        seen = {(endpoint["q"], endpoint["r"])}
        frontier = [(endpoint, 0)]
        candidates = []

        while frontier:
            current, depth = frontier.pop(0)
            if depth >= self.WATER_GATE_RADIUS:
                continue

            for neighbor in tile.neighbors(current).values():
                key = (neighbor["q"], neighbor["r"])
                if key in seen:
                    continue
                seen.add(key)

                info = tile.terrain_at(observation, neighbor)
                if not info or info.get("terrain") == "void":
                    continue

                if info.get("terrain") != "water":
                    enemy_progress = forward_sign * (
                        neighbor["q"] - barrier_q
                    )
                    if enemy_progress > 0.25:
                        candidates.append(
                            (
                                depth + 1,
                                -enemy_progress,
                                self._strategic_move_cost(
                                    observation,
                                    neighbor,
                                ),
                                neighbor["r"],
                                neighbor["q"],
                                neighbor,
                            )
                        )

                frontier.append((neighbor, depth + 1))

        candidates.sort(key=lambda item: item[:-1])
        return [item[-1] for item in candidates[:8]]

    def _update_water_crossing(self, observation):
        """Once a unit has passed the barrier, never pull it back to an edge."""
        if self.water_barrier_crossed or self.major_water_barrier is None:
            return

        here = me.position(observation)
        barrier_q = self.major_water_barrier["barrier_q"]
        forward_sign = self.major_water_barrier["forward_sign"]

        if forward_sign * (here["q"] - barrier_q) > 0.75:
            self.water_barrier_crossed = True

    def _needs_water_bypass(self, observation):
        if self.major_water_barrier is None:
            return False

        self._update_water_crossing(observation)
        return (
            not self.water_barrier_crossed
            and bool(self.water_gate_candidates)
        )

    def _move_via_water_edge(self, observation, target_id=None):
        """
        Move toward the assigned far-side gate using multi-source Dijkstra.

        Footmen receive a soft bonus for ending adjacent to their Shield Wall
        partner. Cavalry simply follows the edge route and later switches to its
        normal Charge logic after crossing.
        """
        if not self._needs_water_bypass(observation):
            return None

        here = me.position(observation)
        route_costs = self._route_cost_map_to_goals(
            observation,
            self.water_gate_candidates,
        )
        allies = list(visible.allies(observation))
        unit_type = me.unit_type(observation)

        best_path = 0
        best_key = None

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(here, path_id)
            route_cost = route_costs.get(
                self._position_key(destination),
                float("inf"),
            )

            if route_cost == float("inf"):
                continue

            nearby_friendlies = sum(
                1
                for ally in allies
                if tile.distance(destination, ally["position"]) <= 1
            )
            terrain_cost = self._strategic_move_cost(
                observation,
                destination,
            )
            step_count = len(paths.decode(path_id))

            shield_value = 0.0
            if unit_type == "footman":
                shield_value = self._shield_wall_value(destination, allies)

            adjusted_route = route_cost - 0.85 * shield_value

            key = (
                adjusted_route,
                nearby_friendlies,
                terrain_cost,
                -step_count,
                1 if path_id == 0 else 0,
                path_id,
            )

            if best_key is None or key < best_key:
                best_key = key
                best_path = path_id

        if best_key is None:
            fallback_goal = self.water_gate_candidates[0]
            return self._move_toward(
                observation,
                fallback_goal,
                target_id=target_id,
            )

        if best_path == 0:
            if target_id is not None:
                return action.stay(target_id, observation)
            return action.stay()

        if target_id is not None:
            return action.move(best_path, target_id, observation)
        return action.move(best_path)

    def _route_cost_map_to_goals(self, observation, goals):
        """Reverse Dijkstra seeded by several acceptable destination tiles."""
        costs = {}
        frontier = []
        serial = 0

        for goal in goals:
            for goal_tile in self._strategic_goal_tiles(observation, goal):
                key = self._position_key(goal_tile)
                if key in costs:
                    continue
                costs[key] = 0.0
                heapq.heappush(
                    frontier,
                    (0.0, serial, goal_tile),
                )
                serial += 1

        while frontier:
            current_cost, _serial, current = heapq.heappop(frontier)
            current_key = self._position_key(current)

            if current_cost != costs.get(current_key):
                continue

            enter_current = self._strategic_move_cost(
                observation,
                current,
            )

            for predecessor in tile.neighbors(current).values():
                if not self._strategically_passable(
                    observation,
                    predecessor,
                ):
                    continue

                predecessor_key = self._position_key(predecessor)
                new_cost = current_cost + enter_current

                if new_cost < costs.get(
                    predecessor_key,
                    float("inf"),
                ):
                    costs[predecessor_key] = new_cost
                    heapq.heappush(
                        frontier,
                        (new_cost, serial, predecessor),
                    )
                    serial += 1

        return costs

    # ==========================================================
    # SEASON 3 V4: HILL-ANCHORED ARCHERS
    # ==========================================================

    def _archer_strategic_hill_action(
        self,
        observation,
        report_position=None,
    ):
        """
        Keep archers on high ground.  With a reported enemy position, prefer a
        hill from which that tile would be inside archer range.  Otherwise use a
        nearby/forward hill and avoid unnecessary moves once a good hill is held.
        """
        hill = self._best_strategic_hill(
            observation,
            report_position=report_position,
        )

        if hill is None:
            return None

        here = me.position(observation)
        if self._same_position(here, hill):
            return action.stay()

        return self._move_toward(
            observation,
            hill,
        )

    def _best_strategic_hill(
        self,
        observation,
        report_position=None,
    ):
        here = me.position(observation)
        center = tile.at_center(observation)
        forward_sign = 1 if me.direction(observation) == 2 else -1

        occupied = {
            self._position_key(unit["position"])
            for unit in [
                *visible.allies(observation),
                *visible.enemies(observation),
            ]
        }

        hills = self._all_hills(observation)
        if not hills:
            return None

        # Keep the search local enough to avoid every archer choosing the same
        # distant landmark, while still allowing movement on the larger map.
        local_hills = [
            hill
            for hill in hills
            if tile.distance(here, hill) <= 10
        ]
        if local_hills:
            hills = local_hills

        best_hill = None
        best_score = float("inf")

        for hill in hills:
            if (
                not self._same_position(hill, here)
                and self._position_key(hill) in occupied
            ):
                continue

            direct_distance = tile.distance(here, hill)
            center_distance = tile.distance(hill, center)
            forward_progress = forward_sign * (
                hill["q"] - here["q"]
            )

            behind_penalty = 0.0
            if forward_progress < 0:
                behind_penalty = 1.5 * abs(forward_progress)

            # Small lateral preference keeps six archers from all drifting to
            # one hill when several equivalent hills exist.
            lateral_penalty = 0.10 * abs(hill["r"] - here["r"])

            if report_position is not None:
                target_distance = tile.distance(
                    hill,
                    report_position,
                )
                firing_penalty = 3.0 * max(
                    0,
                    target_distance - self.ARCHER_RANGE,
                )
                objective_penalty = (
                    firing_penalty
                    + 0.15 * target_distance
                )
            else:
                # Without contact, bias toward useful central high ground rather
                # than a hill deep behind the army.
                objective_penalty = 0.45 * center_distance

            score = (
                0.85 * direct_distance
                + objective_penalty
                + behind_penalty
                + lateral_penalty
                - 0.20 * max(0, forward_progress)
            )

            # A current hill gets persistence so archers do not bounce between
            # nearly equivalent high-ground tiles every activation.
            if self._same_position(hill, here):
                score -= 1.25

            if score < best_score:
                best_score = score
                best_hill = hill

        return best_hill

    def _archer_reposition(
        self,
        observation,
        allies,
        enemies,
        target,
    ):
        """
        Season 3 v4 archer repositioning: move toward another hill that improves
        access to the target instead of stepping onto ordinary ground simply to
        improve range.  Movement between hills can temporarily cross non-hill
        tiles, but the strategic destination remains high ground.
        """
        hill = self._best_strategic_hill(
            observation,
            report_position=target["position"],
        )

        if hill is None:
            return super()._archer_reposition(
                observation,
                allies,
                enemies,
                target,
            )

        here = me.position(observation)
        if self._same_position(here, hill):
            return action.stay(
                target["unit_id"],
                observation,
            )

        return self._move_toward(
            observation,
            hill,
            target_id=target["unit_id"],
        )



class Season4Agent(Season3Agent):
    """
    Season 4 v6: capture-zone strategy + selective search-based planning.

    This layer keeps the Season 3 v5 systems intact:
        * 20-unit squad structure and direct FOCUS/TRACK messaging
        * terrain-aware Dijkstra movement and water-edge bypass
        * hill-anchored archers
        * soft Shield Wall footman pairs
        * deliberate cavalry Charge / reset cycles

    Season 4 adds one capture zone.  Footmen near the objective choose among
    four jobs using a small one-round forecast search:
        HOLD       - enter/occupy the zone and preserve Shield Wall when possible
        SCREEN     - protect the squad archer / rear side of the objective
        INTERCEPT  - meet threats on the enemy-facing side of the objective
        ATTACK     - pursue a visible enemy or current squad FOCUS target

    The chosen job is normally kept for a short commitment period so the unit
    does not oscillate every round.  A major zone-state change can force an
    earlier replan.  Archers support the objective from hills, while cavalry
    patrols outside the zone so it can preserve room for Charge attacks.
    """

    # ----------------------------------------------------------
    # Season 4 capture / search settings
    # ----------------------------------------------------------

    ZONE_PLAN_RADIUS = 8
    ZONE_COMMIT_ROUNDS = 3
    ZONE_FORECAST_HORIZON = 1
    ZONE_CAPTURE_WEIGHT = 6.0
    ZONE_HEALTH_WEIGHT = 0.9
    ZONE_CONTROL_BONUS = 4.0
    ZONE_CONTEST_PENALTY = 1.5
    ZONE_JOB_MESSAGE_TTL = 3

    # Role priors spread the eight footmen instead of making every one choose the
    # center tile. Search can still override these preferences when the forecast
    # strongly favors another job.
    ZONE_ROLE_BIAS = 1.75

    def reset(self, seed, observation) -> None:
        super().reset(seed, observation)

        self.zone_job = None
        self.zone_goal = None
        self.zone_decided_at = -999
        self.zone_state_at_decision = None
        self.zone_job_score = None

        self.reported_zone_jobs = {}
        self.last_sent_zone_job_signature = None
        self.last_sent_zone_job_round = -999

    # ==========================================================
    # SEASON 4 MAIN LAYER
    # ==========================================================

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        zone = self._season4_zone(observation)
        if zone is None:
            return super().act(observation)

        unit_type = me.unit_type(observation)
        allies = list(visible.allies(observation))
        enemies = list(visible.enemies(observation))
        here = me.position(observation)

        # Preserve the proven Season 3 water-edge split before trying to converge
        # on the Season 4 objective. Once the unit is on the far side, normal zone
        # strategy takes over.
        if (
            not enemies
            and unit_type in ("footman", "cavalry")
            and self._needs_water_bypass(observation)
        ):
            return super().act(observation)

        # ------------------------------------------------------
        # FOOTMEN: search-based job selection near the objective.
        # Immediate melee and guaranteed kills still override planning.
        # ------------------------------------------------------
        if unit_type == "footman":
            adjacent = [
                enemy
                for enemy in enemies
                if tile.distance(here, enemy["position"]) <= 1
            ]
            if adjacent:
                return super().act(observation)

            kill_order = self._footman_reachable_kill_order(
                observation,
                allies,
                enemies,
            )
            if kill_order is not None:
                enemy, path_id = kill_order
                self.combat_state = "ZONE_IMMEDIATE_KILL"
                if path_id == 0:
                    return action.stay(enemy["unit_id"], observation)
                return action.move(path_id, enemy["unit_id"], observation)

            if self._should_use_zone_planner(observation, zone, enemies):
                if self._zone_plan_needs_refresh(observation, zone):
                    self.zone_job, self.zone_goal, self.zone_job_score = (
                        self._choose_zone_job(observation, zone, allies, enemies)
                    )
                    self.zone_decided_at = self._current_round(observation)
                    self.zone_state_at_decision = self._zone_visible_state(
                        observation,
                        zone,
                    )
                    self._queue_zone_job_message(
                        observation,
                        self.zone_job,
                        self.zone_goal,
                    )

                order = self._execute_zone_job(
                    observation,
                    zone,
                    self.zone_job,
                    self.zone_goal,
                    allies,
                    enemies,
                )
                if order is not None:
                    return order

        # ------------------------------------------------------
        # ARCHERS: preserve emergency combat, otherwise occupy hills that can
        # cover the capture zone instead of walking into it.
        # ------------------------------------------------------
        if unit_type == "archer" and not enemies:
            zone_hill_order = self._archer_zone_support_action(
                observation,
                zone,
            )
            if zone_hill_order is not None:
                self.combat_state = "ZONE_HILL_SUPPORT"
                return zone_hill_order

        # ------------------------------------------------------
        # CAVALRY: preserve Charge combat. With no visible enemy, patrol outside
        # the objective on the enemy-facing side so a 3+ tile charge remains
        # possible when attackers arrive.
        # ------------------------------------------------------
        if unit_type == "cavalry" and not enemies:
            patrol_order = self._cavalry_zone_patrol_action(
                observation,
                zone,
            )
            if patrol_order is not None:
                self.combat_state = "ZONE_PATROL"
                return patrol_order

        return super().act(observation)

    # ==========================================================
    # SEASON 4 MESSAGING
    # ==========================================================

    def chat(self, inbox: list[dict]) -> list[dict]:
        """
        Keep FOCUS/TRACK messaging and additionally understand:

            ZONEJOB <job> <q> <r> <round>

        This lets squadmates know the footman's current objective.  Because chat
        follows act(), the information can influence only their next activation.
        """
        outgoing = super().chat(inbox)

        for message in inbox:
            sender = message.get("from")
            if sender not in self.allied_player_ids:
                continue

            parts = message.get("text", "").split()
            if len(parts) != 5 or parts[0] != "ZONEJOB":
                continue

            try:
                q = int(parts[2])
                r = int(parts[3])
                report_round = int(parts[4])
            except (TypeError, ValueError):
                continue

            self.reported_zone_jobs[sender] = {
                "job": parts[1],
                "goal": {"q": q, "r": r},
                "round": report_round,
                "tick": self._safe_int(message.get("tick"), -1),
            }

        return outgoing

    def _queue_zone_job_message(self, observation, job, goal):
        if job is None or goal is None:
            return

        current_round = self._current_round(observation)
        signature = (job, goal["q"], goal["r"])

        if (
            signature == self.last_sent_zone_job_signature
            and current_round - self.last_sent_zone_job_round < 2
        ):
            return

        text = f"ZONEJOB {job} {goal['q']} {goal['r']} {current_round}"
        self._queue_direct_to_squad(text)
        self.last_sent_zone_job_signature = signature
        self.last_sent_zone_job_round = current_round

    # ==========================================================
    # CAPTURE-ZONE HELPERS
    # ==========================================================

    def _season4_zone(self, observation):
        raw = observation.get("observation", observation)
        zones = raw.get("battlefield", {}).get("zones", ())
        return zones[0] if zones else None

    def _zone_tiles(self, zone):
        tiles_raw = zone.get("tiles", ()) if zone else ()
        tiles_list = [
            {"q": int(pos["q"]), "r": int(pos["r"])}
            for pos in tiles_raw
            if isinstance(pos, dict) and "q" in pos and "r" in pos
        ]
        if not tiles_list and zone and isinstance(zone.get("center"), dict):
            tiles_list = [zone["center"]]
        return tiles_list

    def _position_in_zone(self, position, zone):
        key = self._position_key(position)
        return any(self._position_key(pos) == key for pos in self._zone_tiles(zone))

    def _zone_visible_state(self, observation, zone):
        own_side = me.side(observation)
        enemy_side = "blue" if own_side == "red" else "red"

        friendly = 1 if self._position_in_zone(me.position(observation), zone) else 0
        enemy = 0

        for unit in visible.allies(observation):
            if self._position_in_zone(unit["position"], zone):
                friendly += 1

        for unit in visible.enemies(observation):
            if self._position_in_zone(unit["position"], zone):
                enemy += 1

        if friendly and enemy:
            label = "contested"
        elif friendly:
            label = "friendly"
        elif enemy:
            label = "enemy"
        else:
            label = "empty"

        raw = observation.get("observation", observation)
        capture = raw.get("capture", {})
        return {
            "label": label,
            "friendly_visible": friendly,
            "enemy_visible": enemy,
            "our_score": int(capture.get(own_side, 0)),
            "enemy_score": int(capture.get(enemy_side, 0)),
        }

    def _should_use_zone_planner(self, observation, zone, enemies):
        here = me.position(observation)
        center = zone.get("center")
        if center is None:
            return False

        if self._position_in_zone(here, zone):
            return True

        if tile.distance(here, center) <= self.ZONE_PLAN_RADIUS:
            return True

        # A visible threat approaching the objective is enough to activate the
        # planner even when the footman itself is slightly farther away.
        if any(tile.distance(enemy["position"], center) <= 5 for enemy in enemies):
            return True

        # Designated capture footmen begin moving toward the objective earlier.
        _unit_type, index = self._unit_type_and_index(self.own_unit_id)
        return index is not None and index <= 3

    def _zone_plan_needs_refresh(self, observation, zone):
        if self.zone_job is None or self.zone_goal is None:
            return True

        current_round = self._current_round(observation)
        if current_round >= self.zone_decided_at + self.ZONE_COMMIT_ROUNDS:
            return True

        current_state = self._zone_visible_state(observation, zone)
        old_state = self.zone_state_at_decision
        if old_state is None:
            return True

        # Major events end the commitment early: the objective becomes contested,
        # changes side, or an enemy appears where none was previously visible.
        if current_state["label"] != old_state.get("label"):
            if "contested" in (current_state["label"], old_state.get("label")):
                return True
            if {current_state["label"], old_state.get("label")} & {"enemy", "friendly"}:
                return True

        return False

    # ==========================================================
    # SEARCH-BASED JOB SELECTION
    # ==========================================================

    def _choose_zone_job(self, observation, zone, allies, enemies):
        candidates = self._zone_job_candidates(
            observation,
            zone,
            allies,
            enemies,
        )

        if not candidates:
            center = zone.get("center") or me.position(observation)
            return "HOLD", center, float("-inf")

        best = None
        best_value = float("-inf")

        for job, goal, order in candidates:
            ending = self._forecast_one_round(observation, order)
            value = self._evaluate_zone_forecast(
                observation,
                zone,
                ending,
                job,
                goal,
                allies,
                enemies,
            )

            if value > best_value:
                best_value = value
                best = (job, goal, value)

        if best is None:
            center = zone.get("center") or me.position(observation)
            return "HOLD", center, float("-inf")

        return best

    def _zone_job_candidates(self, observation, zone, allies, enemies):
        options = []

        # HOLD is always a valid candidate because the objective exists.
        hold_goal = zone.get("center") or me.position(observation)
        hold_order = self._zone_hold_order(
            observation,
            zone,
            allies,
            enemies,
        )
        if hold_order is not None:
            options.append(("HOLD", hold_goal, hold_order))

        screen_goal = self._zone_screen_goal(observation, zone, allies, enemies)
        if screen_goal is not None:
            screen_order = self._move_toward(observation, screen_goal)
            options.append(("SCREEN", screen_goal, screen_order))

        intercept_goal, intercept_target = self._zone_intercept_goal(
            observation,
            zone,
            enemies,
        )
        if intercept_goal is not None:
            intercept_order = self._move_toward(
                observation,
                intercept_goal,
                target_id=(
                    intercept_target["unit_id"]
                    if intercept_target is not None
                    else None
                ),
            )
            options.append(("INTERCEPT", intercept_goal, intercept_order))

        attack_target = self._zone_attack_target(observation, enemies)
        if attack_target is not None:
            attack_order = self._move_toward(
                observation,
                attack_target["position"],
                target_id=attack_target["unit_id"],
            )
            options.append(("ATTACK", attack_target["position"], attack_order))

        return options

    def _forecast_one_round(self, observation, order):
        """
        Forecast until this unit's next activation. Other visible units are
        predicted to stay, which still lets them strike enemies in range.

        This intentionally uses a short horizon because Season 4 has 20 units and
        forecasting every candidate for four rounds on every activation would be
        too expensive for the action time limit.  We copy the small amount of
        forecast state we need before closing the temporary environment.
        """
        trial = None
        try:
            trial = forecast.from_observation(observation)
            own_player = trial.agent_selection
            trial.step(order)

            while trial.agents and trial.agent_selection != own_player:
                _obs, _reward, terminated, truncated, _info = trial.last()
                if terminated or truncated:
                    trial.step(None)
                else:
                    trial.step(action.stay())

            match = trial.match
            unit_snapshots = []
            for unit_id, unit in match.units.items():
                position = getattr(unit, "position", None)
                if isinstance(position, dict):
                    position_copy = {
                        "q": position.get("q"),
                        "r": position.get("r"),
                    }
                else:
                    q = getattr(position, "q", None)
                    r = getattr(position, "r", None)
                    position_copy = (
                        {"q": q, "r": r}
                        if q is not None and r is not None
                        else None
                    )

                unit_snapshots.append(
                    {
                        "unit_id": unit_id,
                        "side": getattr(unit, "side", None),
                        "hit_points": getattr(unit, "hit_points", 0),
                        "position": position_copy,
                    }
                )

            return {
                "capture_scores": dict(getattr(match, "capture_scores", {}) or {}),
                "units": unit_snapshots,
            }
        except Exception:
            # A forecast is advisory, never worth breaking the real action.
            return None
        finally:
            if trial is not None:
                try:
                    trial.close()
                except Exception:
                    pass

    def _evaluate_zone_forecast(
        self,
        observation,
        zone,
        ending,
        job,
        goal,
        allies,
        enemies,
    ):
        # Static utility remains useful if a forecast could not be constructed.
        value = self._zone_job_static_value(
            observation,
            zone,
            job,
            goal,
            allies,
            enemies,
        )

        if ending is None:
            return value

        own_side = me.side(observation)
        enemy_side = "blue" if own_side == "red" else "red"

        # Capture score is the main Season 4 strategic signal.
        capture_scores = ending.get("capture_scores", {}) if isinstance(ending, dict) else {}
        try:
            ours = float(capture_scores.get(own_side, 0))
            theirs = float(capture_scores.get(enemy_side, 0))
            value += self.ZONE_CAPTURE_WEIGHT * (ours - theirs)
        except Exception:
            pass

        # Preserve the searching footman's life; a dead holder cannot keep scoring.
        units_snapshot = ending.get("units", []) if isinstance(ending, dict) else []
        try:
            survivor = next(
                (
                    unit
                    for unit in units_snapshot
                    if unit.get("unit_id") == me.unit_id(observation)
                ),
                None,
            )
            health = float(survivor.get("hit_points", 0)) if survivor is not None else 0.0
            value += self.ZONE_HEALTH_WEIGHT * health
        except Exception:
            pass

        # Add a small predicted control signal from the simulated unit positions.
        try:
            friendly_in_zone = 0
            enemy_in_zone = 0
            zone_keys = {self._position_key(pos) for pos in self._zone_tiles(zone)}

            for unit in units_snapshot:
                pos = unit.get("position")
                side = unit.get("side")
                if pos is None:
                    continue
                key = self._position_key(pos)
                if key not in zone_keys:
                    continue
                if side == own_side:
                    friendly_in_zone += 1
                elif side == enemy_side:
                    enemy_in_zone += 1

            if friendly_in_zone and not enemy_in_zone:
                value += self.ZONE_CONTROL_BONUS
            elif enemy_in_zone and not friendly_in_zone:
                value -= self.ZONE_CONTROL_BONUS
            elif friendly_in_zone and enemy_in_zone:
                value -= self.ZONE_CONTEST_PENALTY
        except Exception:
            pass

        return value

    def _zone_job_static_value(
        self,
        observation,
        zone,
        job,
        goal,
        allies,
        enemies,
    ):
        state = self._zone_visible_state(observation, zone)
        value = 0.0

        # Deterministic role priors spread the eight footmen across complementary
        # jobs, but tactical evidence can outweigh them.
        _unit_type, index = self._unit_type_and_index(self.own_unit_id)
        if index is not None:
            preferred = (
                "HOLD" if index <= 3
                else "SCREEN" if index <= 5
                else "INTERCEPT"
            )
            if job == preferred:
                value += self.ZONE_ROLE_BIAS

        if job == "HOLD":
            if state["label"] in ("empty", "friendly"):
                value += 2.0
            if state["label"] == "enemy":
                value += 1.2
            if state["label"] == "contested":
                value += 1.5

        elif job == "SCREEN":
            archers = [ally for ally in allies if ally.get("type") == "archer"]
            if archers:
                value += 1.0
            cavalry_threats = [
                enemy
                for enemy in enemies
                if enemy.get("type") == "cavalry"
            ]
            if cavalry_threats and archers:
                nearest = min(
                    tile.distance(enemy["position"], archer["position"])
                    for enemy in cavalry_threats
                    for archer in archers
                )
                value += max(0.0, 3.0 - 0.45 * nearest)

        elif job == "INTERCEPT":
            center = zone.get("center")
            if center is not None and enemies:
                nearest = min(
                    tile.distance(enemy["position"], center)
                    for enemy in enemies
                )
                value += max(0.0, 3.5 - 0.45 * nearest)
                if any(enemy.get("type") == "cavalry" for enemy in enemies):
                    value += 0.8

        elif job == "ATTACK":
            if enemies:
                value += 1.5
                weak = min(enemy.get("hit_points", 99) for enemy in enemies)
                value += max(0.0, (6.0 - weak) * 0.15)

        # Shorter route to the job goal is better.
        if goal is not None:
            route_costs = self._route_cost_map(observation, goal)
            cost = route_costs.get(
                self._position_key(me.position(observation)),
                float("inf"),
            )
            if cost != float("inf"):
                value += 2.0 / (1.0 + cost)

        return value

    # ==========================================================
    # EXECUTE THE CHOSEN FOOTMAN JOB
    # ==========================================================

    def _execute_zone_job(
        self,
        observation,
        zone,
        job,
        goal,
        allies,
        enemies,
    ):
        if job == "HOLD":
            self.combat_state = "ZONE_HOLD"
            return self._zone_hold_order(
                observation,
                zone,
                allies,
                enemies,
            )

        if job == "SCREEN":
            self.combat_state = "ZONE_SCREEN"
            dynamic_goal = self._zone_screen_goal(
                observation,
                zone,
                allies,
                enemies,
            )
            return self._move_toward(
                observation,
                dynamic_goal or goal,
            )

        if job == "INTERCEPT":
            self.combat_state = "ZONE_INTERCEPT"
            dynamic_goal, target = self._zone_intercept_goal(
                observation,
                zone,
                enemies,
            )
            chosen_goal = dynamic_goal or goal
            if chosen_goal is None:
                return None
            return self._move_toward(
                observation,
                chosen_goal,
                target_id=(target["unit_id"] if target is not None else None),
            )

        if job == "ATTACK":
            self.combat_state = "ZONE_ATTACK"
            target = self._zone_attack_target(observation, enemies)
            if target is None:
                return None
            return self._move_toward(
                observation,
                target["position"],
                target_id=target["unit_id"],
            )

        return None

    def _zone_hold_order(self, observation, zone, allies, enemies):
        here = me.position(observation)
        zone_tiles = self._zone_tiles(zone)
        if not zone_tiles:
            return None

        # If already inside, hold unless a legal move inside the zone gives a much
        # better Shield Wall / congestion position.
        best_path = None
        best_score = float("inf")
        all_allies = list(visible.allies(observation))
        center = zone.get("center") or zone_tiles[0]

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(here, path_id)
            if not self._position_in_zone(destination, zone):
                continue

            congestion = sum(
                1
                for ally in all_allies
                if tile.distance(destination, ally["position"]) <= 1
            )
            adjacent_enemies = sum(
                1
                for enemy in enemies
                if tile.distance(destination, enemy["position"]) <= 1
            )
            shield = self._shield_wall_value(destination, all_allies)
            center_distance = tile.distance(destination, center)
            terrain_utility = self._terrain_utility(
                observation,
                destination,
                unit_type="footman",
                purpose="defense",
            )

            score = (
                0.40 * center_distance
                + 0.42 * congestion
                + 0.65 * adjacent_enemies
                - 1.25 * shield
                - 0.25 * terrain_utility
            )
            if path_id == 0:
                score -= 0.20

            if score < best_score:
                best_score = score
                best_path = path_id

        if best_path is not None:
            target = self._nearest_visible_enemy(enemies, here)
            if best_path == 0:
                if target is not None:
                    return action.stay(target["unit_id"], observation)
                return action.stay()
            if target is not None:
                return action.move(best_path, target["unit_id"], observation)
            return action.move(best_path)

        # The zone is not reachable this activation. Dijkstra handles the multi-turn
        # route while the action mask keeps the immediate move legal.
        return self._move_toward(observation, center)

    def _zone_screen_goal(self, observation, zone, allies, enemies):
        center = zone.get("center")
        if center is None:
            return None

        squad_archers = [
            ally
            for ally in allies
            if ally.get("type") == "archer"
            and (
                not self.squad_archer_ids
                or ally.get("unit_id") in self.squad_archer_ids
            )
        ]

        if squad_archers:
            archer = min(
                squad_archers,
                key=lambda item: tile.distance(center, item["position"]),
            )
            threats = enemies or []
            if threats:
                threat = min(
                    threats,
                    key=lambda item: (
                        0 if item.get("type") == "cavalry" else 1,
                        tile.distance(item["position"], archer["position"]),
                        item.get("hit_points", 99),
                    ),
                )
                return self._screen_position(
                    archer["position"],
                    threat["position"],
                )
            return self._passable_neighbor_near(
                observation,
                archer["position"],
                center,
            )

        # Without a visible squad archer, guard the rear edge of the objective.
        forward_sign = 1 if me.direction(observation) == 2 else -1
        zone_tiles = self._zone_tiles(zone)
        return min(
            zone_tiles,
            key=lambda pos: (
                forward_sign * pos["q"],
                tile.distance(pos, center),
                pos["r"],
            ),
        ) if zone_tiles else center

    def _zone_intercept_goal(self, observation, zone, enemies):
        center = zone.get("center")
        if center is None:
            return None, None

        if enemies:
            target = min(
                enemies,
                key=lambda enemy: (
                    0 if enemy.get("type") == "cavalry" else 1,
                    tile.distance(enemy["position"], center),
                    enemy.get("hit_points", 99),
                    enemy.get("unit_id", ""),
                ),
            )
            return target["position"], target

        # No visible threat: occupy the enemy-facing edge just outside / at the
        # zone rather than stacking on its center.
        forward_sign = 1 if me.direction(observation) == 2 else -1
        zone_tiles = self._zone_tiles(zone)
        if not zone_tiles:
            return center, None

        front = max(
            zone_tiles,
            key=lambda pos: (
                forward_sign * pos["q"],
                -tile.distance(pos, center),
                -pos["r"],
            ),
        )

        # Prefer a passable neighbor one step farther toward the enemy if possible.
        candidates = []
        for neighbor in tile.neighbors(front).values():
            if not self._strategically_passable(observation, neighbor):
                continue
            progress = forward_sign * (neighbor["q"] - front["q"])
            candidates.append((progress, -self._strategic_move_cost(observation, neighbor), neighbor))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2], None
        return front, None

    def _zone_attack_target(self, observation, enemies):
        if not enemies:
            return None

        for enemy in enemies:
            if enemy.get("unit_id") == self.reported_focus_target_id:
                return enemy

        here = me.position(observation)
        return min(
            enemies,
            key=lambda enemy: (
                tile.distance(here, enemy["position"]),
                enemy.get("hit_points", 99),
                enemy.get("unit_id", ""),
            ),
        )

    def _nearest_visible_enemy(self, enemies, from_position):
        if not enemies:
            return None
        return min(
            enemies,
            key=lambda enemy: (
                tile.distance(from_position, enemy["position"]),
                enemy.get("hit_points", 99),
                enemy.get("unit_id", ""),
            ),
        )

    def _passable_neighbor_near(self, observation, anchor, toward):
        candidates = []
        for neighbor in tile.neighbors(anchor).values():
            if not self._strategically_passable(observation, neighbor):
                continue
            candidates.append(
                (
                    tile.distance(neighbor, toward),
                    self._strategic_move_cost(observation, neighbor),
                    neighbor["q"],
                    neighbor["r"],
                    neighbor,
                )
            )
        if not candidates:
            return anchor
        candidates.sort(key=lambda item: item[:-1])
        return candidates[0][-1]

    # ==========================================================
    # ARCHER / CAVALRY OBJECTIVE SUPPORT
    # ==========================================================

    def _archer_zone_support_action(self, observation, zone):
        center = zone.get("center")
        if center is None:
            return None

        # Reuse the Season 3 hill chooser, but make the capture zone the position
        # the hill should cover. This keeps archers out of the crowded objective.
        hill = self._best_strategic_hill(
            observation,
            report_position=center,
        )
        if hill is None:
            return None

        here = me.position(observation)
        if self._same_position(here, hill):
            return action.stay()
        return self._move_toward(observation, hill)

    def _cavalry_zone_patrol_action(self, observation, zone):
        center = zone.get("center")
        if center is None:
            return None

        here = me.position(observation)
        goal = self._cavalry_zone_patrol_goal(observation, zone)
        if goal is None:
            return None

        # If already at a useful 3-5 tile patrol distance, hold the flank instead
        # of entering the capture zone and losing room for the next Charge.
        center_distance = tile.distance(here, center)
        if 3 <= center_distance <= 5 and tile.distance(here, goal) <= 1:
            return action.stay()

        return self._move_toward(observation, goal)

    def _cavalry_zone_patrol_goal(self, observation, zone):
        center = zone.get("center")
        if center is None:
            return None

        forward_sign = 1 if me.direction(observation) == 2 else -1
        candidates = []

        # Search a small ring around the zone for open, enemy-facing tiles.
        seen = {self._position_key(center)}
        frontier = [(center, 0)]
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= 4:
                continue

            for neighbor in tile.neighbors(current).values():
                key = self._position_key(neighbor)
                if key in seen:
                    continue
                seen.add(key)
                nd = depth + 1

                if self._strategically_passable(observation, neighbor):
                    zone_distance = tile.distance(neighbor, center)
                    if 3 <= zone_distance <= 4:
                        info = tile.terrain_at(observation, neighbor) or {}
                        feature = info.get("feature")
                        progress = forward_sign * (neighbor["q"] - center["q"])
                        marsh_penalty = 2 if feature == "marsh" else 0
                        forest_penalty = 1 if feature == "forest" else 0
                        candidates.append(
                            (
                                -progress,
                                marsh_penalty,
                                forest_penalty,
                                self._strategic_move_cost(observation, neighbor),
                                neighbor["r"],
                                neighbor["q"],
                                neighbor,
                            )
                        )

                frontier.append((neighbor, nd))

        if not candidates:
            return center

        candidates.sort(key=lambda item: item[:-1])
        return candidates[0][-1]


class Agent(Season4Agent):
    """
    Season 5 v7 teacher: three-zone strategic allocation.

    This class intentionally remains a hand-authored teacher.  It keeps the full
    Season 4/Season 3 tactical stack below it, but replaces the single-objective
    layer with three-zone reasoning so its decisions can later be collected as
    imitation-learning demonstrations.

    High-level plan:
        * Footmen are the primary zone holders/capturers.
        * Archers support assigned zones from nearby hills and still use the
          inherited emergency retreat behavior whenever enemies are visible.
        * Cavalry moves between objectives quickly, pressures weak zones, patrols
          friendly objectives, and keeps the inherited Charge/reset combat logic.
        * Each unit has a deterministic default objective so the army starts
          spread across all three zones instead of converging on one.
        * Local observations and direct squad reports can override the default
          objective when another zone becomes contested or enemy-controlled.

    New direct-message protocol:
        ZONE3STATE <zone_index> <label> <friendly> <enemy> <round>
        ZONE3JOB   <job> <zone_index> <q> <r> <round>

    Distant zones are treated as UNKNOWN rather than EMPTY.  This is important:
    an agent cannot infer that a capture zone is empty simply because no occupants
    are inside its current vision.
    """

    # ==========================================================
    # SEASON 5 THREE-ZONE SETTINGS
    # ==========================================================

    THREE_ZONE_COMMIT_ROUNDS = 3
    THREE_ZONE_REPORT_TTL = 3
    THREE_ZONE_STATE_REPEAT_ROUNDS = 2
    THREE_ZONE_JOB_REPEAT_ROUNDS = 2

    # How close the acting unit must be to a zone center before its own
    # observation is considered authoritative for occupancy.
    UNIT_VISION_RADIUS = {
        "footman": 4,
        "archer": 6,
        "cavalry": 6,
    }
    ZONE_OBSERVATION_MARGIN = 2

    # A pair of footmen assigned to a friendly home zone should normally stay.
    # This prevents the "we own one zone, everyone leaves it to attack another"
    # failure mode.
    FRIENDLY_ZONE_MIN_VISIBLE_DEFENDERS = 2

    # Strategic weights.  These are intentionally readable because this file is
    # the teacher whose demonstrations will later be imitated.
    ZONE_URGENCY = {
        "contested": 10.0,
        "enemy": 8.5,
        "empty": 6.0,
        "unknown": 5.2,
        "friendly": 2.0,
    }

    FOOTMAN_HOME_BONUS = 2.4
    ARCHER_HOME_BONUS = 1.8
    CAVALRY_HOME_BONUS = 1.2

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(self, seed, observation) -> None:
        super().reset(seed, observation)

        # Season 5 commitment.  The inherited Season 4 one-zone memory is left in
        # place for compatibility, but this is the state used by the v7 teacher.
        self.three_zone_job = None
        self.three_zone_index = None
        self.three_zone_goal = None
        self.three_zone_decided_at = -999
        self.three_zone_state_at_decision = None
        self.three_zone_score = None

        # Freshest direct squad reports for each objective.
        self.reported_three_zone_states = {}
        self.reported_three_zone_jobs = {}

        self.last_sent_zone3_state = {}
        self.last_sent_zone3_job_signature = None
        self.last_sent_zone3_job_round = -999

    # ==========================================================
    # MAIN SEASON 5 LAYER
    # ==========================================================

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        zones = self._season5_zones(observation)

        # The same file can still run in the old one-zone environment.
        if len(zones) < 3:
            return Season4Agent.act(self, observation)

        unit_type = me.unit_type(observation)
        allies = list(visible.allies(observation))
        enemies = list(visible.enemies(observation))
        here = me.position(observation)

        # Share only zone states this unit can genuinely observe.  Messages are
        # direct to squadmates, so the enemy does not receive the objective report.
        self._queue_observed_zone_state_updates(observation, zones)

        # Preserve the proven water bypass before strategic convergence.
        if (
            not enemies
            and unit_type in ("footman", "cavalry")
            and self._needs_water_bypass(observation)
        ):
            self.combat_state = "S5_WATER_EDGE_BYPASS"
            order = self._move_via_water_edge(observation)
            if order is not None:
                return order

        # ------------------------------------------------------
        # ARCHERS
        # ------------------------------------------------------
        # If anything is visible, the inherited combat tree keeps emergency
        # cavalry retreat, hill fighting, kill shots, and target selection.
        if unit_type == "archer":
            if enemies:
                return Season3Agent.act(self, observation)

            zone_index, zone_score = self._choose_three_zone_for_role_committed(
                observation,
                zones,
                role="archer",
            )
            zone = zones[zone_index]
            self._remember_simple_three_zone_job(
                observation,
                "SUPPORT_ZONE",
                zone_index,
                zone.get("center") or here,
                zone_score,
            )
            order = self._archer_three_zone_action(
                observation,
                zone_index,
                zone,
            )
            if order is not None:
                self.combat_state = f"S5_SUPPORT_ZONE_{zone_index}"
                return order
            return Season3Agent.act(self, observation)

        # ------------------------------------------------------
        # CAVALRY
        # ------------------------------------------------------
        # Visible contact still uses the full Charge/reset combat policy.
        if unit_type == "cavalry":
            if enemies:
                return Season3Agent.act(self, observation)

            zone_index, zone_score = self._choose_three_zone_for_role_committed(
                observation,
                zones,
                role="cavalry",
            )
            zone = zones[zone_index]
            state = self._effective_zone_state(
                observation,
                zones,
                zone_index,
            )

            if state["label"] in ("enemy", "empty", "unknown"):
                job = "PRESSURE_ZONE"
            elif state["label"] == "contested":
                job = "REINFORCE_ZONE"
            else:
                job = "PATROL_ZONE"

            self._remember_simple_three_zone_job(
                observation,
                job,
                zone_index,
                zone.get("center") or here,
                zone_score,
            )

            order = self._cavalry_three_zone_action(
                observation,
                zone_index,
                zone,
                state,
            )
            if order is not None:
                self.combat_state = f"S5_{job}_{zone_index}"
                return order
            return Season3Agent.act(self, observation)

        # ------------------------------------------------------
        # FOOTMEN
        # ------------------------------------------------------
        if unit_type == "footman":
            # Immediate melee remains more important than strategy.
            adjacent = [
                enemy
                for enemy in enemies
                if tile.distance(here, enemy["position"]) <= 1
            ]
            if adjacent:
                return Season3Agent.act(self, observation)

            # A guaranteed kill is also worth taking before changing objectives.
            kill_order = self._footman_reachable_kill_order(
                observation,
                allies,
                enemies,
            )
            if kill_order is not None:
                enemy, path_id = kill_order
                self.combat_state = "S5_IMMEDIATE_KILL"
                if path_id == 0:
                    return action.stay(enemy["unit_id"], observation)
                return action.move(path_id, enemy["unit_id"], observation)

            if self._three_zone_plan_needs_refresh(observation, zones):
                (
                    self.three_zone_job,
                    self.three_zone_index,
                    self.three_zone_goal,
                    self.three_zone_score,
                ) = self._choose_three_zone_footman_job(
                    observation,
                    zones,
                    allies,
                    enemies,
                )
                self.three_zone_decided_at = self._current_round(observation)
                self.three_zone_state_at_decision = self._effective_zone_state(
                    observation,
                    zones,
                    self.three_zone_index,
                )
                self._queue_three_zone_job_message(
                    observation,
                    self.three_zone_job,
                    self.three_zone_index,
                    self.three_zone_goal,
                )

            order = self._execute_three_zone_footman_job(
                observation,
                zones,
                self.three_zone_job,
                self.three_zone_index,
                self.three_zone_goal,
                allies,
                enemies,
            )
            if order is not None:
                return order

        return Season3Agent.act(self, observation)

    # ==========================================================
    # SEASON 5 PEER MESSAGING
    # ==========================================================

    def chat(self, inbox: list[dict]) -> list[dict]:
        """
        Preserve Season 3 FOCUS/TRACK handling and add three-zone reports.

        We intentionally call Season3Agent.chat directly.  The inherited Season 4
        parser expects the old one-zone ZONEJOB format, while Season 5 carries an
        explicit zone index.
        """
        outgoing = Season3Agent.chat(self, inbox)

        for message in inbox:
            sender = message.get("from")
            if sender not in self.allied_player_ids:
                continue

            parts = message.get("text", "").split()
            tick = self._safe_int(message.get("tick"), -1)

            # ZONE3STATE <idx> <label> <friendly> <enemy> <round>
            if len(parts) == 6 and parts[0] == "ZONE3STATE":
                try:
                    zone_index = int(parts[1])
                    label = parts[2]
                    friendly = int(parts[3])
                    enemy = int(parts[4])
                    report_round = int(parts[5])
                except (TypeError, ValueError):
                    continue

                if zone_index < 0 or zone_index >= 3:
                    continue
                if label not in ("friendly", "enemy", "contested", "empty"):
                    continue

                old = self.reported_three_zone_states.get(zone_index)
                new_key = (report_round, tick)
                old_key = (
                    old.get("round", -999),
                    old.get("tick", -1),
                ) if old is not None else (-999, -1)

                if new_key >= old_key:
                    self.reported_three_zone_states[zone_index] = {
                        "label": label,
                        "friendly_visible": friendly,
                        "enemy_visible": enemy,
                        "round": report_round,
                        "tick": tick,
                        "sender": sender,
                    }
                continue

            # ZONE3JOB <job> <idx> <q> <r> <round>
            if len(parts) == 6 and parts[0] == "ZONE3JOB":
                try:
                    zone_index = int(parts[2])
                    q = int(parts[3])
                    r = int(parts[4])
                    report_round = int(parts[5])
                except (TypeError, ValueError):
                    continue

                if zone_index < 0 or zone_index >= 3:
                    continue

                self.reported_three_zone_jobs[sender] = {
                    "job": parts[1],
                    "zone_index": zone_index,
                    "goal": {"q": q, "r": r},
                    "round": report_round,
                    "tick": tick,
                }

        return outgoing

    def _queue_observed_zone_state_updates(self, observation, zones):
        current_round = self._current_round(observation)

        for zone_index, zone in enumerate(zones):
            if not self._zone_observed_locally(
                observation,
                zone,
            ):
                continue

            state = self._zone_visible_state(observation, zone)
            signature = (
                state["label"],
                state["friendly_visible"],
                state["enemy_visible"],
            )
            old = self.last_sent_zone3_state.get(zone_index)

            if (
                old is not None
                and old.get("signature") == signature
                and current_round - old.get("round", -999)
                < self.THREE_ZONE_STATE_REPEAT_ROUNDS
            ):
                continue

            text = (
                f"ZONE3STATE {zone_index} {state['label']} "
                f"{state['friendly_visible']} {state['enemy_visible']} "
                f"{current_round}"
            )
            self._queue_direct_to_squad(text)
            self.last_sent_zone3_state[zone_index] = {
                "signature": signature,
                "round": current_round,
            }

    def _queue_three_zone_job_message(
        self,
        observation,
        job,
        zone_index,
        goal,
    ):
        if job is None or goal is None or zone_index is None:
            return

        current_round = self._current_round(observation)
        signature = (
            job,
            zone_index,
            goal["q"],
            goal["r"],
        )

        if (
            signature == self.last_sent_zone3_job_signature
            and current_round - self.last_sent_zone3_job_round
            < self.THREE_ZONE_JOB_REPEAT_ROUNDS
        ):
            return

        text = (
            f"ZONE3JOB {job} {zone_index} "
            f"{goal['q']} {goal['r']} {current_round}"
        )
        self._queue_direct_to_squad(text)
        self.last_sent_zone3_job_signature = signature
        self.last_sent_zone3_job_round = current_round

    # ==========================================================
    # THREE-ZONE PERCEPTION
    # ==========================================================

    def _season5_zones(self, observation):
        raw = observation.get("observation", observation)
        zones = list(raw.get("battlefield", {}).get("zones", ()) or ())
        return zones[:3]

    def _zone_observed_locally(self, observation, zone):
        center = zone.get("center") if zone else None
        if center is None:
            return False

        unit_type = me.unit_type(observation)
        radius = self.UNIT_VISION_RADIUS.get(unit_type, 4)
        return tile.distance(
            me.position(observation),
            center,
        ) <= radius + self.ZONE_OBSERVATION_MARGIN

    def _effective_zone_state(self, observation, zones, zone_index):
        zone = zones[zone_index]
        local = self._zone_visible_state(observation, zone)

        # If the objective is actually in/near this unit's vision, its observation
        # overrides all delayed messages.
        if self._zone_observed_locally(observation, zone):
            result = dict(local)
            result["source"] = "local"
            return result

        report = self.reported_three_zone_states.get(zone_index)
        if report is not None:
            age = self._current_round(observation) - report.get("round", -999)
            if 0 <= age <= self.THREE_ZONE_REPORT_TTL:
                result = {
                    "label": report.get("label", "unknown"),
                    "friendly_visible": report.get("friendly_visible", 0),
                    "enemy_visible": report.get("enemy_visible", 0),
                    "our_score": local.get("our_score", 0),
                    "enemy_score": local.get("enemy_score", 0),
                    "source": "message",
                }
                return result

        # No evidence means unknown, not empty.
        return {
            "label": "unknown",
            "friendly_visible": 0,
            "enemy_visible": 0,
            "our_score": local.get("our_score", 0),
            "enemy_score": local.get("enemy_score", 0),
            "source": "unknown",
        }

    def _all_effective_zone_states(self, observation, zones):
        return [
            self._effective_zone_state(observation, zones, idx)
            for idx in range(len(zones))
        ]

    def _zone_index_containing(self, position, zones):
        for idx, zone in enumerate(zones):
            if self._position_in_zone(position, zone):
                return idx
        return None

    # ==========================================================
    # DETERMINISTIC INITIAL ALLOCATION
    # ==========================================================

    def _default_zone_index(self, unit_type=None):
        parsed_type, index = self._unit_type_and_index(self.own_unit_id)
        if unit_type is None:
            unit_type = parsed_type

        if index is None:
            return self.squad_id % 3

        if unit_type == "footman":
            # Shield Wall pairs 0/1, 2/3, and 4/5 each get one objective.
            # Pair 6/7 is the mobile reserve and therefore has no hard home.
            pair = index // 2
            if pair <= 2:
                return pair
            return None

        if unit_type in ("archer", "cavalry"):
            return index % 3

        return index % 3

    def _footman_pair_index(self):
        unit_type, index = self._unit_type_and_index(self.own_unit_id)
        if unit_type != "footman" or index is None:
            return None
        return index // 2

    # ==========================================================
    # ZONE SCORING / REALLOCATION
    # ==========================================================

    def _zone_need_score(
        self,
        observation,
        zones,
        zone_index,
        role,
        home_zone,
    ):
        zone = zones[zone_index]
        state = self._effective_zone_state(
            observation,
            zones,
            zone_index,
        )
        center = zone.get("center")
        here = me.position(observation)

        score = self.ZONE_URGENCY.get(state["label"], 5.0)

        # Visible enemy presence makes an objective more urgent.
        score += 1.15 * state.get("enemy_visible", 0)

        # A friendly zone with too few visible defenders needs reinforcement.
        if state["label"] == "friendly":
            friendly = state.get("friendly_visible", 0)
            if friendly < self.FRIENDLY_ZONE_MIN_VISIBLE_DEFENDERS:
                score += 2.4
            elif friendly >= 4:
                score -= 1.4

        # Contested objectives are worth a little more if our side already has
        # somebody present: reinforcement may flip the point immediately.
        if (
            state["label"] == "contested"
            and state.get("friendly_visible", 0) > 0
        ):
            score += 1.0

        # Stable role allocation prevents all 20 units from choosing the same
        # currently attractive objective.
        if home_zone is not None and zone_index == home_zone:
            if role == "footman":
                score += self.FOOTMAN_HOME_BONUS
            elif role == "archer":
                score += self.ARCHER_HOME_BONUS
            elif role == "cavalry":
                score += self.CAVALRY_HOME_BONUS

        # Prefer objectives that can be reached efficiently.  Reverse Dijkstra is
        # used so water, marsh, hills, and forest costs remain consistent with the
        # rest of the agent.
        if center is not None:
            route_costs = self._route_cost_map(observation, center)
            route_cost = route_costs.get(
                self._position_key(here),
                float("inf"),
            )
            if route_cost != float("inf"):
                role_scale = (
                    0.12 if role == "cavalry"
                    else 0.17 if role == "archer"
                    else 0.20
                )
                score -= role_scale * route_cost
            else:
                score -= 4.0

        return score

    def _choose_three_zone_for_role(self, observation, zones, role):
        home_zone = self._default_zone_index(role)

        scored = []
        for zone_index in range(len(zones)):
            score = self._zone_need_score(
                observation,
                zones,
                zone_index,
                role,
                home_zone,
            )
            scored.append((score, -zone_index, zone_index))

        scored.sort(reverse=True)
        return scored[0][2], scored[0][0]

    def _choose_three_zone_for_role_committed(self, observation, zones, role):
        """
        Keep archer/cavalry assignments stable for a few rounds unless another
        objective becomes contested.  This mirrors the footman commitment idea and
        prevents support units from oscillating between similarly scored zones.
        """
        current_round = self._current_round(observation)

        valid_existing = (
            self.three_zone_index is not None
            and 0 <= self.three_zone_index < len(zones)
            and self.three_zone_goal is not None
        )

        if valid_existing:
            age = current_round - self.three_zone_decided_at
            current_state = self._effective_zone_state(
                observation,
                zones,
                self.three_zone_index,
            )

            urgent_elsewhere = False
            for idx in range(len(zones)):
                if idx == self.three_zone_index:
                    continue
                state = self._effective_zone_state(
                    observation,
                    zones,
                    idx,
                )
                if (
                    state["label"] == "contested"
                    and current_state["label"] != "contested"
                ):
                    urgent_elsewhere = True
                    break

            if (
                0 <= age < self.THREE_ZONE_COMMIT_ROUNDS
                and not urgent_elsewhere
            ):
                return (
                    self.three_zone_index,
                    self.three_zone_score
                    if self.three_zone_score is not None
                    else 0.0,
                )

        return self._choose_three_zone_for_role(
            observation,
            zones,
            role,
        )

    def _choose_three_zone_footman_job(
        self,
        observation,
        zones,
        allies,
        enemies,
    ):
        unit_type, index = self._unit_type_and_index(self.own_unit_id)
        home_zone = self._default_zone_index("footman")
        pair_index = self._footman_pair_index()
        here = me.position(observation)
        current_zone = self._zone_index_containing(here, zones)

        # A non-reserve Shield Wall pair should defend its home objective when that
        # objective is already friendly.  This leaves a durable pair behind while
        # the other pairs and reserve can change jobs elsewhere.
        if home_zone is not None:
            home_state = self._effective_zone_state(
                observation,
                zones,
                home_zone,
            )
            if (
                home_state["label"] == "friendly"
                and current_zone == home_zone
            ):
                goal = zones[home_zone].get("center") or here
                return (
                    "HOLD_ZONE",
                    home_zone,
                    goal,
                    100.0,
                )

        scored = []
        for zone_index, zone in enumerate(zones):
            state = self._effective_zone_state(
                observation,
                zones,
                zone_index,
            )
            score = self._zone_need_score(
                observation,
                zones,
                zone_index,
                "footman",
                home_zone,
            )

            center = zone.get("center")
            if center is None:
                continue

            # If this footman is already inside a friendly objective, staying is
            # valuable when there are not enough defenders.
            if current_zone == zone_index and state["label"] == "friendly":
                if (
                    state.get("friendly_visible", 0)
                    <= self.FRIENDLY_ZONE_MIN_VISIBLE_DEFENDERS
                ):
                    score += 5.0
                else:
                    score += 1.0

            # Mobile reserve pair 6/7 deliberately follows the hottest objective.
            if pair_index == 3:
                if state["label"] in ("contested", "enemy"):
                    score += 2.0
                elif state["label"] == "friendly":
                    score -= 0.8

            scored.append((score, -zone_index, zone_index, state))

        if not scored:
            goal = me.position(observation)
            return "HOLD_ZONE", 0, goal, float("-inf")

        scored.sort(reverse=True)
        best_score, _neg_idx, zone_index, state = scored[0]
        zone = zones[zone_index]
        center = zone.get("center") or here

        # Visible enemies near the selected objective make interception sensible.
        near_enemies = [
            enemy
            for enemy in enemies
            if tile.distance(enemy["position"], center) <= 4
        ]

        if state["label"] == "friendly":
            if near_enemies:
                job = "INTERCEPT_ZONE"
            else:
                job = "HOLD_ZONE"

        elif state["label"] == "contested":
            job = "REINFORCE_ZONE"

        elif state["label"] in ("enemy", "empty", "unknown"):
            # If we can already see a defender near the objective, call the job an
            # attack/capture rather than blindly walking to the center.
            job = "CAPTURE_ZONE"

        else:
            job = "CAPTURE_ZONE"

        return job, zone_index, center, best_score

    # ==========================================================
    # COMMITMENT / REPLANNING
    # ==========================================================

    def _three_zone_plan_needs_refresh(self, observation, zones):
        if (
            self.three_zone_job is None
            or self.three_zone_index is None
            or self.three_zone_goal is None
        ):
            return True

        if self.three_zone_index >= len(zones):
            return True

        current_round = self._current_round(observation)
        current_state = self._effective_zone_state(
            observation,
            zones,
            self.three_zone_index,
        )

        old_state = self.three_zone_state_at_decision or {}
        old_label = old_state.get("label")
        new_label = current_state.get("label")

        # Major objective changes end the commitment immediately.
        if old_label != new_label:
            if "contested" in (old_label, new_label):
                return True
            if {old_label, new_label} & {"friendly", "enemy"}:
                return True
            if old_label == "unknown" and new_label != "unknown":
                return True

        # Do not keep a capture assignment after the objective becomes safely ours.
        if (
            self.three_zone_job in ("CAPTURE_ZONE", "REINFORCE_ZONE")
            and new_label == "friendly"
        ):
            return True

        # Reconsider after the normal commitment window.
        if (
            current_round
            >= self.three_zone_decided_at + self.THREE_ZONE_COMMIT_ROUNDS
        ):
            return True

        # A different zone becoming contested is strategically urgent enough to
        # reconsider even before the timer expires, unless we are currently holding
        # a friendly home objective.
        home_zone = self._default_zone_index("footman")
        holding_home = (
            self.three_zone_job == "HOLD_ZONE"
            and self.three_zone_index == home_zone
            and new_label == "friendly"
        )
        if not holding_home:
            for idx in range(len(zones)):
                if idx == self.three_zone_index:
                    continue
                state = self._effective_zone_state(
                    observation,
                    zones,
                    idx,
                )
                if state["label"] == "contested":
                    return True

        return False

    def _remember_simple_three_zone_job(
        self,
        observation,
        job,
        zone_index,
        goal,
        score,
    ):
        current_round = self._current_round(observation)
        signature = (
            job,
            zone_index,
            goal["q"],
            goal["r"],
        )

        old_signature = (
            self.three_zone_job,
            self.three_zone_index,
            self.three_zone_goal["q"] if self.three_zone_goal else None,
            self.three_zone_goal["r"] if self.three_zone_goal else None,
        )

        self.three_zone_job = job
        self.three_zone_index = zone_index
        self.three_zone_goal = goal
        self.three_zone_score = score

        if signature != old_signature:
            self.three_zone_decided_at = current_round
            self._queue_three_zone_job_message(
                observation,
                job,
                zone_index,
                goal,
            )

    # ==========================================================
    # FOOTMAN JOB EXECUTION
    # ==========================================================

    def _execute_three_zone_footman_job(
        self,
        observation,
        zones,
        job,
        zone_index,
        goal,
        allies,
        enemies,
    ):
        if zone_index is None or zone_index >= len(zones):
            return None

        zone = zones[zone_index]
        center = zone.get("center") or goal

        if job == "HOLD_ZONE":
            self.combat_state = f"S5_HOLD_ZONE_{zone_index}"
            return self._zone_hold_order(
                observation,
                zone,
                allies,
                enemies,
            )

        if job in ("CAPTURE_ZONE", "REINFORCE_ZONE"):
            self.combat_state = f"S5_{job}_{zone_index}"

            # Once inside, use the inherited holder logic to choose a good Shield
            # Wall tile and attack nearby enemies while preserving objective control.
            if self._position_in_zone(me.position(observation), zone):
                return self._zone_hold_order(
                    observation,
                    zone,
                    allies,
                    enemies,
                )

            # If a visible defender is close to this objective, engage it rather
            # than walking past it to the geometric center.
            target = self._enemy_near_zone(
                enemies,
                center,
                max_distance=3,
            )
            if target is not None:
                return self._footman_active_attack(
                    observation,
                    allies,
                    enemies,
                    target,
                )

            return self._footman_move_to_zone(
                observation,
                zone,
                allies,
            )

        if job == "INTERCEPT_ZONE":
            self.combat_state = f"S5_INTERCEPT_ZONE_{zone_index}"
            intercept_goal, target = self._zone_intercept_goal(
                observation,
                zone,
                enemies,
            )
            chosen_goal = intercept_goal or center
            return self._move_toward(
                observation,
                chosen_goal,
                target_id=(
                    target["unit_id"]
                    if target is not None
                    else None
                ),
            )

        return self._footman_move_to_zone(
            observation,
            zone,
            allies,
        )

    def _footman_move_to_zone(self, observation, zone, allies):
        """
        Move toward an objective with the same soft Shield Wall preference used by
        the Season 3 advance code.
        """
        center = zone.get("center")
        if center is None:
            return None

        here = me.position(observation)
        route_costs = self._route_cost_map(
            observation,
            center,
        )
        all_allies = list(visible.allies(observation))

        best_path = 0
        best_score = float("inf")

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(here, path_id)
            route_cost = route_costs.get(
                self._position_key(destination),
                float("inf"),
            )
            if route_cost == float("inf"):
                continue

            shield = self._shield_wall_value(
                destination,
                all_allies,
            )
            congestion = sum(
                1
                for ally in all_allies
                if tile.distance(destination, ally["position"]) <= 1
            )
            terrain = self._terrain_utility(
                observation,
                destination,
                unit_type="footman",
                purpose="mobility",
            )
            step_count = len(paths.decode(path_id))

            score = (
                route_cost
                + 0.28 * congestion
                - 1.10 * shield
                - 0.20 * terrain
                - 0.05 * step_count
            )

            # Entering the target objective is strategically valuable.
            if self._position_in_zone(destination, zone):
                score -= 1.50

            if path_id == 0 and not self._position_in_zone(here, zone):
                score += 0.30

            if score < best_score:
                best_score = score
                best_path = path_id

        if best_path == 0:
            if self._position_in_zone(here, zone):
                return action.stay()
            return self._move_toward(observation, center)

        return action.move(best_path)

    def _enemy_near_zone(self, enemies, center, max_distance):
        if center is None:
            return None
        candidates = [
            enemy
            for enemy in enemies
            if tile.distance(enemy["position"], center) <= max_distance
        ]
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda enemy: (
                0 if enemy.get("unit_id") == self.reported_focus_target_id else 1,
                0 if enemy.get("type") == "cavalry" else 1,
                enemy.get("hit_points", 99),
                tile.distance(enemy["position"], center),
                enemy.get("unit_id", ""),
            ),
        )

    # ==========================================================
    # ARCHER THREE-ZONE SUPPORT
    # ==========================================================

    def _archer_three_zone_action(
        self,
        observation,
        zone_index,
        zone,
    ):
        center = zone.get("center")
        if center is None:
            return None

        here = me.position(observation)
        distance = tile.distance(here, center)

        # When far from the assigned objective, first make strategic progress.
        # Once close enough, the inherited hill chooser finds high ground that can
        # actually cover the zone.
        if distance > self.ARCHER_RANGE + 3:
            return self._archer_approach_zone(
                observation,
                zone,
            )

        return self._archer_zone_support_action(
            observation,
            zone,
        )

    def _archer_approach_zone(self, observation, zone):
        center = zone.get("center")
        if center is None:
            return None

        here = me.position(observation)
        route_costs = self._route_cost_map(
            observation,
            center,
        )
        best_path = 0
        best_score = float("inf")

        for path_id in action.legal_paths(observation):
            destination = tile.at_path_end(here, path_id)
            route_cost = route_costs.get(
                self._position_key(destination),
                float("inf"),
            )
            if route_cost == float("inf"):
                continue

            terrain = self._terrain_utility(
                observation,
                destination,
                unit_type="archer",
                purpose="mobility",
            )
            on_hill = self._is_hill(observation, destination)
            in_zone = self._position_in_zone(destination, zone)
            step_count = len(paths.decode(path_id))

            score = (
                route_cost
                - 0.55 * terrain
                - (0.85 if on_hill else 0.0)
                - 0.05 * step_count
                + (1.20 if in_zone else 0.0)
            )

            if score < best_score:
                best_score = score
                best_path = path_id

        if best_path == 0:
            return action.stay()
        return action.move(best_path)

    # ==========================================================
    # CAVALRY THREE-ZONE SUPPORT
    # ==========================================================

    def _cavalry_three_zone_action(
        self,
        observation,
        zone_index,
        zone,
        state,
    ):
        center = zone.get("center")
        if center is None:
            return None

        here = me.position(observation)
        distance = tile.distance(here, center)

        # Fast cavalry can secure a genuinely observed empty objective if nobody
        # else is there.  For UNKNOWN objectives it approaches but does not assume
        # the center is safe.
        if state["label"] == "empty":
            if not self._position_in_zone(here, zone):
                return self._move_toward(
                    observation,
                    center,
                )
            return action.stay()

        # Enemy/unknown objectives: close rapidly until the cavalry is near enough
        # to see defenders; visible contact will switch to Charge combat next act.
        if state["label"] in ("enemy", "unknown"):
            if distance > 3:
                return self._move_toward(
                    observation,
                    center,
                )
            patrol = self._cavalry_zone_patrol_action(
                observation,
                zone,
            )
            return patrol or self._move_toward(
                observation,
                center,
            )

        # Contested zones need immediate reinforcement.  Enter if not already in
        # the objective; otherwise hold long enough to deny the enemy point.
        if state["label"] == "contested":
            if not self._position_in_zone(here, zone):
                return self._move_toward(
                    observation,
                    center,
                )
            return action.stay()

        # Friendly zone: stay outside the capture tiles when possible so cavalry
        # has room to launch a >=3-hex Charge if attackers arrive.
        return self._cavalry_zone_patrol_action(
            observation,
            zone,
        )
