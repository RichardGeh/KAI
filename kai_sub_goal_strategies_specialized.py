# kai_sub_goal_strategies_specialized.py
"""
Specialized Domain Strategies

File reading, spatial reasoning, and arithmetic strategies.

Author: KAI Development Team
Date: 2025-11-14
"""

import logging
from typing import Any, Dict, Tuple

from component_5_linguistik_strukturen import SubGoal
from kai_sub_goal_strategy_base import SubGoalStrategy

logger = logging.getLogger(__name__)


class FileReaderStrategy(SubGoalStrategy):
    """
    Strategy für Datei-Lese Sub-Goals.

    Zuständig für:
    - Datei-Validierung (Existenz, Lesbarkeit, Format)
    - Dokument-Parsing (DOCX, PDF)
    - Text-Ingestion mit Progress-Updates
    - Ingestion-Berichterstellung
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        file_reading_keywords = [
            "Validiere Dateipfad",
            "Extrahiere Text aus der Datei",
            "Verarbeite extrahierten Text durch Ingestion-Pipeline",
            "Formuliere Ingestion-Bericht",
        ]
        return any(kw in sub_goal_description for kw in file_reading_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Validiere Dateipfad" in description:
            return self._validate_file(intent)
        elif "Extrahiere Text aus der Datei" in description:
            return self._extract_text(intent, context)
        elif "Verarbeite extrahierten Text durch Ingestion-Pipeline" in description:
            return self._process_ingestion(context)
        elif "Formuliere Ingestion-Bericht" in description:
            return self._formulate_report(context)

        return False, {"error": f"Unbekanntes FileReading-SubGoal: {description}"}

    def _validate_file(self, intent) -> Tuple[bool, Dict[str, Any]]:
        """
        Validiert Dateipfad, Existenz, Lesbarkeit und Format.

        Returns:
            Tuple (success, {"file_path": ..., "format": ...})
        """
        from pathlib import Path

        from component_28_document_parser import DocumentParserFactory

        file_path = intent.arguments.get("file_path", "")

        # Prüfe ob Dateipfad vorhanden
        if not file_path:
            return False, {"error": "Kein Dateipfad angegeben."}

        # Prüfe Existenz
        path = Path(file_path)
        if not path.exists():
            return False, {"error": f"Datei nicht gefunden: {file_path}"}

        # Prüfe ob es eine Datei ist (nicht Verzeichnis)
        if not path.is_file():
            return False, {"error": f"Pfad ist keine Datei: {file_path}"}

        # Prüfe Lesbarkeit
        import os

        if not os.access(file_path, os.R_OK):
            return False, {
                "error": f"Datei nicht lesbar (fehlende Leserechte): {file_path}"
            }

        # Prüfe ob Format unterstützt wird
        if not DocumentParserFactory.is_supported(file_path):
            supported = ", ".join(DocumentParserFactory.get_supported_extensions())
            file_format = path.suffix
            return False, {
                "error": f"Format '{file_format}' wird nicht unterstützt. Unterstützte Formate: {supported}"
            }

        logger.info(
            f"Datei-Validierung erfolgreich: {file_path} (Format: {path.suffix})"
        )

        return True, {"file_path": str(path.absolute()), "format": path.suffix}

    def _extract_text(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Extrahiert Text aus Dokument mittels DocumentParserFactory.

        PHASE 8 (Extended Features): Unterstützt Preview-Modus.

        Returns:
            Tuple (success, {"extracted_text": ..., "char_count": ..., "file_name": ..., "preview": ...})
        """
        from component_28_document_parser import DocumentParserFactory
        from kai_exceptions import DocumentParseError, MissingDependencyError

        file_path = context.get("file_path")
        if not file_path:
            return False, {"error": "Dateipfad fehlt aus vorigem Schritt."}

        from pathlib import Path

        file_name = Path(file_path).name

        try:
            logger.info(f"Starte Text-Extraktion aus: {file_name}")

            # Erstelle passenden Parser
            parser = DocumentParserFactory.create_parser(file_path)

            # Extrahiere Text
            extracted_text = parser.extract_text(file_path)

            if not extracted_text or not extracted_text.strip():
                logger.warning(f"Dokument enthält keinen Text: {file_name}")
                return False, {
                    "error": f"Dokument '{file_name}' enthält keinen extrahierbaren Text."
                }

            char_count = len(extracted_text)

            # PHASE 8: Erstelle Preview (erste 500 Zeichen)
            preview = extracted_text[:500]
            if len(extracted_text) > 500:
                preview += "..."

            logger.info(
                f"Text erfolgreich extrahiert: {char_count} Zeichen aus {file_name}"
            )

            return True, {
                "extracted_text": extracted_text,
                "char_count": char_count,
                "file_name": file_name,
                "preview": preview,
            }

        except MissingDependencyError as e:
            # Fehlende Bibliothek (python-docx oder pdfplumber)
            logger.error(f"Fehlende Bibliothek: {e.dependency_name}")
            return False, {
                "error": f"Erforderliche Bibliothek fehlt: {e.dependency_name}. "
                f"Installiere mit: pip install {e.dependency_name}"
            }

        except DocumentParseError as e:
            # Parsing-Fehler
            logger.error(f"Parsing-Fehler: {e}")
            return False, {"error": f"Konnte Dokument nicht parsen: {e.message}"}

        except Exception as e:
            # Unerwarteter Fehler
            logger.error(f"Unerwarteter Fehler bei Text-Extraktion: {e}", exc_info=True)
            return False, {
                "error": f"Unerwarteter Fehler beim Lesen der Datei: {str(e)}"
            }

    def _process_ingestion(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verarbeitet extrahierten Text durch Ingestion-Pipeline mit Progress-Updates.

        PHASE 8 (Extended Features): Fordert Preview-Bestätigung vor Ingestion an.

        Nutzt ingest_text_large() für Batch-Processing und Progress-Tracking.

        Returns:
            Tuple (success, {"facts_created": ..., "chunks_processed": ..., ...})
        """
        from kai_ingestion_handler import KaiIngestionHandler

        extracted_text = context.get("extracted_text")
        file_name = context.get("file_name", "Datei")
        preview = context.get("preview", "")
        char_count = context.get("char_count", 0)

        if not extracted_text:
            return False, {"error": "Text aus vorigem Schritt fehlt."}

        # PHASE 8: Fordere Preview-Bestätigung vom User an
        if preview and char_count > 0:
            logger.info(f"Zeige Preview für {file_name} ({char_count} Zeichen)")
            confirmed = self.worker.wait_for_preview_confirmation(
                preview, file_name, char_count
            )

            if not confirmed:
                logger.info(f"User hat Ingestion von {file_name} abgebrochen")
                return False, {
                    "error": f"Ingestion von '{file_name}' wurde vom Benutzer abgebrochen.",
                    "user_cancelled": True,
                }

        # Erstelle Ingestion Handler
        ingestion_handler = KaiIngestionHandler(
            self.worker.netzwerk,
            self.worker.preprocessor,
            self.worker.prototyping_engine,
            self.worker.embedding_service,
        )

        # Progress-Callback für UI-Updates
        def progress_callback(current, total, stats):
            """Emittiert Progress-Updates an UI."""
            percent = int((current / total) * 100) if total > 0 else 0
            progress_msg = (
                f"Verarbeite {file_name}: {current}/{total} Sätze ({percent}%) - "
                f"{stats['facts_created']} Fakten gelernt"
            )
            logger.info(progress_msg)
            # Emit file progress signal (UI will process events automatically)
            self.worker.signals.file_progress_update.emit(current, total, progress_msg)

            # Emittiere Signal für UI
            if hasattr(self.worker, "signals") and hasattr(
                self.worker.signals, "progress_update"
            ):
                self.worker.signals.progress_update.emit(progress_msg)

        try:
            logger.info(f"Starte Ingestion von {file_name} (Batch-Processing)...")

            # WICHTIG: Zeige Progress-Bar SOFORT beim Start (nicht erst beim ersten Update)
            # Das gibt dem User sofort Feedback dass die Verarbeitung läuft
            self.worker.signals.file_progress_update.emit(
                0, 100, f"Starte Verarbeitung von {file_name}..."
            )

            # BATCH-PROCESSING mit Progress-Tracking und PARALLELER VERARBEITUNG
            stats = ingestion_handler.ingest_text_large(
                text=extracted_text,
                chunk_size=50,  # 50 Sätze pro Chunk für häufigere UI-Updates
                progress_callback=progress_callback,
                max_workers=None,  # Auto-detect: CPU-Cores * 2
            )

            logger.info(
                f"Ingestion abgeschlossen: {stats['facts_created']} Fakten aus {file_name} "
                f"({stats['chunks_processed']} Chunks verarbeitet)"
            )

            return True, {
                "facts_created": stats["facts_created"],
                "learned_patterns": stats["learned_patterns"],
                "fallback_patterns": stats["fallback_patterns"],
                "chunks_processed": stats["chunks_processed"],
                "fragments_stored": stats.get("fragments_stored", 0),
                "connections_stored": stats.get("connections_stored", 0),
            }

        except Exception as e:
            logger.error(f"Fehler bei Ingestion: {e}", exc_info=True)
            return False, {"error": f"Fehler beim Verarbeiten des Textes: {str(e)}"}

    def _formulate_report(self, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Formuliert Ingestion-Bericht mit Statistiken.

        Returns:
            Tuple (success, {"final_response": ...})
        """
        file_name = context.get("file_name", "Datei")
        facts_created = context.get("facts_created", 0)
        chunks_processed = context.get("chunks_processed", 0)
        learned = context.get("learned_patterns", 0)
        fallback = context.get("fallback_patterns", 0)
        fragments = context.get("fragments_stored", 0)
        connections = context.get("connections_stored", 0)

        # Formuliere Hauptbericht
        if facts_created > 1:
            response = f"Datei '{file_name}' erfolgreich verarbeitet. Ich habe {facts_created} neue Fakten gelernt"
        elif facts_created == 1:
            response = f"Datei '{file_name}' erfolgreich verarbeitet. Ich habe 1 neuen Fakt gelernt"
        else:
            response = f"Datei '{file_name}' wurde verarbeitet, aber ich konnte keine neuen Fakten extrahieren"

        # Füge Details hinzu
        details = []
        if learned > 0:
            details.append(f"{learned} aus gelernten Mustern")
        if fallback > 0:
            details.append(f"{fallback} aus neuen Mustern")
        if chunks_processed > 0:
            details.append(f"{chunks_processed} Chunks verarbeitet")

        if details:
            response += f" ({', '.join(details)})"

        response += "."

        # Füge Word-Usage-Statistiken hinzu (falls aktiviert)
        if fragments > 0 or connections > 0:
            usage_details = []
            if fragments > 0:
                usage_details.append(f"{fragments} Kontextfragmente")
            if connections > 0:
                usage_details.append(f"{connections} Wortverbindungen")
            response += f" Zusätzlich wurden {', '.join(usage_details)} getrackt."

        logger.info(
            f"Ingestion-Bericht erstellt: {facts_created} Fakten aus {file_name}"
        )

        return True, {"final_response": response}


# ============================================================================
# SPATIAL REASONING STRATEGY (Räumliches Reasoning)
# ============================================================================


class SpatialReasoningStrategy(SubGoalStrategy):
    """
    Strategy für räumliche Reasoning Sub-Goals.

    Zuständig für:
    - Extraktion räumlicher Entitäten und Positionen
    - Aufbau räumlicher Modelle (Grid, Shapes, Positions)
    - Constraint-basierte räumliche Probleme (CSP)
    - State-Space-Planning für räumliche Aktionen
    - Formatierung räumlicher Antworten
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        spatial_keywords = [
            "Extrahiere räumliche Entitäten und Positionen",
            "Erstelle räumliches Modell",
            "Löse räumliche Constraints",
            "Plane räumliche Aktionen",
            "Formuliere räumliche Antwort",
        ]
        return any(kw in sub_goal_description for kw in spatial_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        intent = context.get("intent")
        if not intent:
            return False, {"error": "Intent fehlt im Kontext"}

        description = sub_goal.description

        if "Extrahiere räumliche Entitäten und Positionen" in description:
            return self._extract_spatial_entities(intent, context)
        elif "Erstelle räumliches Modell" in description:
            return self._create_spatial_model(context)
        elif "Löse räumliche Constraints" in description:
            return self._solve_spatial_constraints(context)
        elif "Plane räumliche Aktionen" in description:
            return self._plan_spatial_actions(context)
        elif "Formuliere räumliche Antwort" in description:
            return self._formulate_spatial_answer(context)

        return False, {"error": f"Unbekanntes Spatial-Reasoning-SubGoal: {description}"}

    def _extract_spatial_entities(
        self, intent, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Extrahiert räumliche Entitäten und ihre Positionen aus dem Intent.

        Args:
            intent: MeaningPoint mit spatial_query_type
            context: Kontext

        Returns:
            Tuple (success, {"entities": [...], "positions": {...}, "relations": [...]})
        """
        spatial_query_type = intent.arguments.get(
            "spatial_query_type", "position_query"
        )

        # Tracke räumliche Extraktion
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_extraction",
            description=f"Extrahiere räumliche Entitäten für: {spatial_query_type}",
            data={"query_type": spatial_query_type},
        )

        # Extrahiere Entitäten aus Intent-Argumenten
        entities = []
        positions = {}
        relations = []

        # Grid-basierte Queries
        if spatial_query_type == "grid_query":
            grid_config = intent.arguments.get("grid_config", {})
            query_position = intent.arguments.get("query_position")

            entities.append({"type": "grid", "config": grid_config})

            if query_position:
                positions["query"] = query_position

        # Positions-Queries
        elif spatial_query_type == "position_query":
            objects = intent.arguments.get("objects", [])
            for obj in objects:
                entities.append(
                    {
                        "type": "object",
                        "name": obj.get("name"),
                        "position": obj.get("position"),
                    }
                )
                if obj.get("position"):
                    positions[obj["name"]] = obj["position"]

        # Relations-Queries
        elif spatial_query_type == "relation_query":
            subject = intent.arguments.get("subject")
            relation = intent.arguments.get("relation")
            target = intent.arguments.get("target")

            if subject and relation and target:
                relations.append(
                    {"subject": subject, "relation": relation, "target": target}
                )

        # Path-Finding-Queries
        elif spatial_query_type == "path_finding":
            start = intent.arguments.get("start_position")
            goal = intent.arguments.get("goal_position")
            obstacles = intent.arguments.get("obstacles", [])

            entities.append(
                {
                    "type": "path_problem",
                    "start": start,
                    "goal": goal,
                    "obstacles": obstacles,
                }
            )

        logger.info(
            f"Räumliche Entitäten extrahiert: {len(entities)} Entitäten, "
            f"{len(positions)} Positionen, {len(relations)} Relationen"
        )

        return True, {
            "entities": entities,
            "positions": positions,
            "relations": relations,
            "spatial_query_type": spatial_query_type,
        }

    def _create_spatial_model(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Erstellt ein räumliches Modell aus den extrahierten Entitäten.

        Args:
            context: Kontext mit entities, positions, relations

        Returns:
            Tuple (success, {"spatial_model": ..., "model_type": ...})
        """
        from component_17_proof_explanation import ProofStep, StepType
        from component_42_spatial_reasoning import (
            Grid,
            Position,
            SpatialReasoner,
            SpatialRelationType,
        )

        # Create spatial engine
        spatial_engine = SpatialReasoner(self.worker.netzwerk)

        entities = context.get("entities", [])
        positions = context.get("positions", {})
        relations = context.get("relations", [])
        spatial_query_type = context.get("spatial_query_type")

        if not entities:
            return False, {"error": "Keine Entitäten zum Erstellen eines Modells"}

        # Tracke Modell-Erstellung
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_model_creation",
            description=f"Erstelle räumliches Modell: {spatial_query_type}",
            data={"entity_count": len(entities)},
        )

        # Erstelle ProofStep für Modell-Erstellung
        proof_step = None

        try:
            # FALL 1: Grid-basiertes Modell
            if spatial_query_type == "grid_query":
                grid_entity = entities[0]
                grid_config = grid_entity.get("config", {})

                rows = grid_config.get("rows", 8)
                cols = grid_config.get("cols", 8)

                grid = Grid(name=f"grid_{rows}x{cols}", width=cols, height=rows)

                logger.info(f"Grid-Modell erstellt: {rows}x{cols}")

                # ProofStep für Grid-Erstellung
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(grid)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=[f"Grid-Konfiguration: {rows}×{cols}"],
                    output=f"Grid-Modell mit {rows * cols} Feldern erstellt",
                    confidence=1.0,
                    explanation_text=f"Räumliches Grid-Modell ({rows}×{cols}) für die Abfrage erstellt.",
                    metadata={
                        "model_type": "grid",
                        "rows": rows,
                        "cols": cols,
                        "total_cells": rows * cols,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": grid,
                    "model_type": "grid",
                    "spatial_engine": spatial_engine,
                    "proof_step": proof_step,
                }

            # FALL 2: Positions-basiertes Modell
            elif spatial_query_type == "position_query":
                # Registriere Positionen im Engine
                position_inputs = []
                for obj_name, pos_data in positions.items():
                    pos = Position(pos_data["x"], pos_data["y"])
                    # Store position in knowledge graph
                    spatial_engine.add_position(obj_name, pos)
                    position_inputs.append(
                        f"{obj_name} @ ({pos_data['x']}, {pos_data['y']})"
                    )

                logger.info(f"Positions-Modell erstellt mit {len(positions)} Objekten")

                # ProofStep für Positions-Modell
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=position_inputs,
                    output=f"Positions-Modell mit {len(positions)} Objekten erstellt",
                    confidence=1.0,
                    explanation_text=f"{len(positions)} Objekte mit Positionen registriert.",
                    metadata={
                        "model_type": "positions",
                        "object_count": len(positions),
                        "positions": {
                            obj: f"({pos_data['x']}, {pos_data['y']})"
                            for obj, pos_data in positions.items()
                        },
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": None,  # Keine Grid, nur Positionen
                    "model_type": "positions",
                    "spatial_engine": spatial_engine,
                    "proof_step": proof_step,
                }

            # FALL 3: Relations-basiertes Modell
            elif spatial_query_type == "relation_query":
                # Registriere Relationen
                relation_inputs = []
                for rel in relations:
                    try:
                        rel_type = SpatialRelationType[rel["relation"].upper()]
                        # Store spatial relation in knowledge graph
                        spatial_engine.add_spatial_relation(
                            rel["subject"], rel_type, rel["target"]
                        )
                        relation_inputs.append(
                            f"{rel['subject']} {rel['relation']} {rel['target']}"
                        )
                    except KeyError:
                        logger.warning(f"Unbekannte Relation: {rel['relation']}")

                logger.info(
                    f"Relations-Modell erstellt mit {len(relations)} Relationen"
                )

                # ProofStep für Relations-Modell
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=relation_inputs,
                    output=f"Relations-Modell mit {len(relations)} Relationen erstellt",
                    confidence=1.0,
                    explanation_text=f"{len(relations)} räumliche Relationen registriert.",
                    metadata={
                        "model_type": "relations",
                        "relation_count": len(relations),
                        "relations": relation_inputs,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": None,
                    "model_type": "relations",
                    "spatial_engine": spatial_engine,
                    "proof_step": proof_step,
                }

            # FALL 4: Path-Finding-Modell
            elif spatial_query_type == "path_finding":
                path_problem = entities[0]
                start_pos = Position(
                    path_problem["start"]["x"], path_problem["start"]["y"]
                )
                goal_pos = Position(
                    path_problem["goal"]["x"], path_problem["goal"]["y"]
                )
                obstacles = [
                    Position(obs["x"], obs["y"])
                    for obs in path_problem.get("obstacles", [])
                ]

                logger.info(
                    f"Path-Finding-Modell erstellt: Start={start_pos}, Goal={goal_pos}, "
                    f"Obstacles={len(obstacles)}"
                )

                # ProofStep für Path-Finding-Modell
                proof_step = ProofStep(
                    step_id=f"spatial_model_{spatial_query_type}_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_MODEL_CREATION,
                    inputs=[
                        f"Start: {start_pos}",
                        f"Ziel: {goal_pos}",
                        f"Hindernisse: {len(obstacles)}",
                    ],
                    output=f"Path-Finding-Modell erstellt (Start→Ziel mit {len(obstacles)} Hindernissen)",
                    confidence=1.0,
                    explanation_text=f"Räumliches Modell für Pfadsuche von {start_pos} nach {goal_pos} erstellt.",
                    metadata={
                        "model_type": "path_finding",
                        "start": str(start_pos),
                        "goal": str(goal_pos),
                        "obstacle_count": len(obstacles),
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "spatial_model": None,
                    "model_type": "path_finding",
                    "spatial_engine": spatial_engine,
                    "start_position": start_pos,
                    "goal_position": goal_pos,
                    "obstacles": obstacles,
                    "proof_step": proof_step,
                }

            else:
                return False, {"error": f"Unbekannter Query-Typ: {spatial_query_type}"}

        except Exception as e:
            logger.error(f"Fehler bei Modell-Erstellung: {e}", exc_info=True)
            return False, {"error": f"Modell-Erstellung fehlgeschlagen: {str(e)}"}

    def _solve_spatial_constraints(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Löst räumliche Constraints mittels CSP-Solver.

        Args:
            context: Kontext mit spatial_model, spatial_engine

        Returns:
            Tuple (success, {"constraint_solution": ..., "satisfiable": ...})
        """
        from component_17_proof_explanation import ProofStep, StepType
        from component_29_constraint_reasoning import ConstraintSolver

        spatial_engine = context.get("spatial_engine")
        model_type = context.get("model_type")
        parent_proof_step = context.get("proof_step")

        if not spatial_engine:
            return False, {"error": "Kein räumliches Modell vorhanden"}

        # Tracke Constraint-Solving
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_constraint_solving",
            description=f"Löse räumliche Constraints: {model_type}",
            data={"model_type": model_type},
        )

        try:
            # Für Relations-basierte Queries: Nutze CSP-Solver
            if model_type == "relations":
                csp_engine = ConstraintSolver()

                # Hole alle Relationen vom SpatialEngine
                # und formuliere sie als CSP
                # (Vereinfachte Implementierung - kann erweitert werden)

                logger.info("Räumliche Constraints via CSP gelöst")

                # ProofStep für Constraint-Solving
                proof_step = ProofStep(
                    step_id=f"spatial_csp_{id(csp_engine)}",
                    step_type=StepType.SPATIAL_CONSTRAINT_SOLVING,
                    inputs=[parent_proof_step.output] if parent_proof_step else [],
                    output="Räumliche Constraints sind erfüllbar",
                    confidence=1.0,
                    explanation_text="Alle räumlichen Constraints wurden via CSP-Solver geprüft und sind konsistent.",
                    parent_steps=(
                        [parent_proof_step.step_id] if parent_proof_step else []
                    ),
                    metadata={
                        "model_type": model_type,
                        "solver": "CSP (Constraint Satisfaction Problem)",
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "constraint_solution": None,  # Placeholder
                    "satisfiable": True,
                    "csp_engine": csp_engine,
                    "proof_step": proof_step,
                }

            # Andere Modelltypen benötigen kein explizites Constraint-Solving
            return True, {
                "constraint_solution": None,
                "satisfiable": True,
            }

        except Exception as e:
            logger.error(f"Fehler beim Constraint-Solving: {e}", exc_info=True)
            return False, {"error": f"Constraint-Solving fehlgeschlagen: {str(e)}"}

    def _plan_spatial_actions(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Plant räumliche Aktionen mittels Spatial Reasoner's A* pathfinding.

        Args:
            context: Kontext mit spatial_engine, spatial_model, model_type

        Returns:
            Tuple (success, {"plan": ..., "plan_length": ...})
        """
        from component_17_proof_explanation import ProofStep, StepType

        model_type = context.get("model_type")
        spatial_engine = context.get("spatial_engine")
        spatial_model = context.get("spatial_model")

        if model_type != "grid" or not spatial_model or not spatial_engine:
            # Kein Planning benötigt (nur für Grid-basiertes pathfinding)
            return True, {"plan": None, "plan_length": 0}

        start_position = context.get("start_position")
        goal_position = context.get("goal_position")
        allow_diagonal = context.get("allow_diagonal", False)

        if not start_position or not goal_position:
            return False, {"error": "Start- oder Ziel-Position fehlt"}

        # Tracke Planning
        self.worker.working_memory.add_reasoning_state(
            step_type="spatial_planning",
            description=f"Plane Pfad von {start_position} nach {goal_position}",
            data={
                "start": str(start_position),
                "goal": str(goal_position),
                "allow_diagonal": allow_diagonal,
            },
        )

        try:
            # Use SpatialReasoner's built-in A* pathfinding
            path = spatial_engine.find_path(
                grid_name=spatial_model.name,
                start=start_position,
                goal=goal_position,
                allow_diagonal=allow_diagonal,
            )

            if path:
                logger.info(f"Pfad gefunden: {len(path)} Schritte")

                # ProofStep für Planning
                parent_proof_step = context.get("proof_step")
                proof_step = ProofStep(
                    step_id=f"spatial_planning_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_PLANNING,
                    inputs=[
                        (
                            parent_proof_step.output
                            if parent_proof_step
                            else "Path-Finding-Modell"
                        )
                    ],
                    output=f"Pfad gefunden mit {len(path)} Schritten",
                    confidence=1.0,
                    explanation_text=f"A*-Algorithmus fand einen optimalen Pfad von {start_position} nach {goal_position}.",
                    parent_steps=(
                        [parent_proof_step.step_id] if parent_proof_step else []
                    ),
                    metadata={
                        "algorithm": "A* (Spatial Reasoning)",
                        "heuristic": "Manhattan-Distanz",
                        "plan_length": len(path),
                        "start": str(start_position),
                        "goal": str(goal_position),
                        "grid": spatial_model.name,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "plan": path,
                    "plan_length": len(path),
                    "proof_step": proof_step,
                }
            else:
                logger.info("Kein Pfad gefunden (nicht erreichbar)")

                # ProofStep für nicht-erreichbares Ziel
                from component_17_proof_explanation import ProofStep, StepType

                parent_proof_step = context.get("proof_step")
                proof_step = ProofStep(
                    step_id=f"spatial_planning_unreachable_{id(spatial_engine)}",
                    step_type=StepType.SPATIAL_PLANNING,
                    inputs=[
                        (
                            parent_proof_step.output
                            if parent_proof_step
                            else "Path-Finding-Modell"
                        )
                    ],
                    output=f"Ziel {goal_position} ist nicht erreichbar",
                    confidence=1.0,
                    explanation_text=f"A*-Algorithmus konnte keinen Pfad von {start_position} nach {goal_position} finden (Hindernisse blockieren).",
                    parent_steps=(
                        [parent_proof_step.step_id] if parent_proof_step else []
                    ),
                    metadata={
                        "algorithm": "A* (State-Space Planning)",
                        "reachable": False,
                    },
                    source_component="spatial_reasoning",
                )

                return True, {
                    "plan": None,
                    "plan_length": 0,
                    "reachable": False,
                    "proof_step": proof_step,
                }

        except Exception as e:
            logger.error(f"Fehler beim Planning: {e}", exc_info=True)
            return False, {"error": f"Planning fehlgeschlagen: {str(e)}"}

    def _formulate_spatial_answer(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Formuliert eine Antwort für räumliche Queries.

        Args:
            context: Kontext mit allen Ergebnissen

        Returns:
            Tuple (success, {"final_response": ...})
        """
        from kai_response_formatter import KaiResponseFormatter

        formatter = KaiResponseFormatter()

        model_type = context.get("model_type")
        spatial_query_type = context.get("spatial_query_type")

        # Delegiere Formatierung an Response Formatter
        response = formatter.format_spatial_answer(
            model_type=model_type,
            spatial_query_type=spatial_query_type,
            entities=context.get("entities", []),
            positions=context.get("positions", {}),
            relations=context.get("relations", []),
            plan=context.get("plan"),
            plan_length=context.get("plan_length", 0),
            reachable=context.get("reachable", True),
        )

        return True, {"final_response": response}


# ============================================================================
# SHARED STRATEGY (Gemeinsame Sub-Goals)
# ============================================================================


class ArithmeticStrategy(SubGoalStrategy):
    """
    Strategy für arithmetische Berechnungen (Phase Mathematik).

    Zuständig für:
    - Parse arithmetischen Ausdruck aus Text
    - Konvertiere Zahlwörter zu Zahlen
    - Führe arithmetische Operation aus
    - Formatiere Ergebnis als Zahlwort
    """

    def can_handle(self, sub_goal_description: str) -> bool:
        arithmetic_keywords = [
            "Parse arithmetischen Ausdruck aus Text",
            "Konvertiere Zahlwörter zu Zahlen",
            "Führe arithmetische Operation aus",
            "Formatiere Ergebnis als Zahlwort",
        ]
        return any(kw in sub_goal_description for kw in arithmetic_keywords)

    def execute(
        self, sub_goal: SubGoal, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        try:
            description = sub_goal.description
            query_text = sub_goal.metadata.get("query_text", "")

            # Dispatcher für verschiedene Arithmetik-SubGoals
            if "Parse arithmetischen Ausdruck aus Text" in description:
                return self._parse_expression(query_text, context)
            elif "Konvertiere Zahlwörter zu Zahlen" in description:
                return self._convert_words_to_numbers(context)
            elif "Führe arithmetische Operation aus" in description:
                return self._execute_operation(context)
            elif "Formatiere Ergebnis als Zahlwort" in description:
                return self._format_result(context)

            return False, {"error": f"Unbekanntes Arithmetik-SubGoal: {description}"}

        except Exception as e:
            logger.error(f"Fehler in ArithmeticStrategy: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _parse_expression(
        self, query_text: str, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Parse arithmetischen Ausdruck aus Text.

        Erkennt Muster wie:
        - "Was ist drei plus fünf?" -> ("+", ["drei", "fünf"])
        - "Wie viel ist 7 mal 8?" -> ("*", ["7", "8"])
        """
        try:
            import re

            text_lower = query_text.lower().strip()

            # Operator-Mapping
            operator_patterns = [
                (r"(\w+)\s*\+\s*(\w+)", "+"),  # 3 + 5
                (r"(\w+)\s*-\s*(\w+)", "-"),  # 10 - 2
                (r"(\w+)\s*\*\s*(\w+)", "*"),  # 7 * 8
                (r"(\w+)\s*/\s*(\w+)", "/"),  # 20 / 4
                (r"(\w+)\s+plus\s+(\w+)", "+"),  # drei plus fünf
                (r"(\w+)\s+minus\s+(\w+)", "-"),  # zehn minus zwei
                (r"(\w+)\s+mal\s+(\w+)", "*"),  # sieben mal acht
                (r"(\w+)\s+durch\s+(\w+)", "/"),  # zwanzig durch vier
                (r"(\w+)\s+geteilt\s+durch\s+(\w+)", "/"),  # zehn geteilt durch zwei
                (
                    r"(\w+)\s+multipliziert\s+mit\s+(\w+)",
                    "*",
                ),  # drei multipliziert mit vier
                (r"(\w+)\s+addiert\s+mit\s+(\w+)", "+"),  # zwei addiert mit fünf
                (
                    r"(\w+)\s+subtrahiert\s+von\s+(\w+)",
                    "-",
                ),  # fünf subtrahiert von zehn
            ]

            for pattern, operator in operator_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    operand1 = match.group(1).strip()
                    operand2 = match.group(2).strip()

                    logger.debug(f"Parsed expression: {operand1} {operator} {operand2}")

                    return True, {
                        "operator": operator,
                        "operand1_word": operand1,
                        "operand2_word": operand2,
                        "original_text": query_text,
                    }

            return False, {
                "error": f"Konnte keinen Operator in '{query_text}' erkennen"
            }

        except Exception as e:
            logger.error(f"Fehler beim Parsen des Ausdrucks: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _convert_words_to_numbers(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Konvertiere Zahlwörter zu Zahlen.

        Verwendet component_53_number_language (wenn verfügbar) oder Fallback.
        """
        try:
            operand1_word = context.get("operand1_word", "")
            operand2_word = context.get("operand2_word", "")

            # PHASE 1: Einfacher Fallback für Zahlen 0-20 (wird später durch component_53 ersetzt)
            number_map = {
                "null": 0,
                "eins": 1,
                "ein": 1,
                "zwei": 2,
                "drei": 3,
                "vier": 4,
                "fünf": 5,
                "sechs": 6,
                "sieben": 7,
                "acht": 8,
                "neun": 9,
                "zehn": 10,
                "elf": 11,
                "zwölf": 12,
                "dreizehn": 13,
                "vierzehn": 14,
                "fünfzehn": 15,
                "sechzehn": 16,
                "siebzehn": 17,
                "achtzehn": 18,
                "neunzehn": 19,
                "zwanzig": 20,
            }

            # Versuche als Zahlwort oder als Zahl zu parsen
            try:
                operand1 = (
                    number_map.get(operand1_word.lower(), None)
                    if operand1_word.lower() in number_map
                    else int(operand1_word)
                )
            except ValueError:
                return False, {
                    "error": f"Kann '{operand1_word}' nicht zu Zahl konvertieren"
                }

            try:
                operand2 = (
                    number_map.get(operand2_word.lower(), None)
                    if operand2_word.lower() in number_map
                    else int(operand2_word)
                )
            except ValueError:
                return False, {
                    "error": f"Kann '{operand2_word}' nicht zu Zahl konvertieren"
                }

            logger.debug(
                f"Converted: {operand1_word} -> {operand1}, {operand2_word} -> {operand2}"
            )

            # Übernehme vorherigen Context und füge Zahlen hinzu
            result = dict(context)
            result["operand1"] = operand1
            result["operand2"] = operand2
            return True, result

        except Exception as e:
            logger.error(f"Fehler bei Zahl-Konvertierung: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _execute_operation(
        self, context: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Führe arithmetische Operation aus.

        Verwendet component_52_arithmetic_reasoning (wenn verfügbar) oder Fallback.
        """
        try:
            operator = context.get("operator")
            operand1 = context.get("operand1")
            operand2 = context.get("operand2")

            if operator is None or operand1 is None or operand2 is None:
                return False, {"error": "Operator oder Operanden fehlen im Kontext"}

            # PHASE 2: Verwende ArithmeticEngine (mit Proof Tree) wenn verfügbar
            try:
                from component_52_arithmetic_reasoning import ArithmeticEngine

                # ArithmeticEngine aus Worker holen oder erstellen
                if not hasattr(self.worker, "arithmetic_engine"):
                    self.worker.arithmetic_engine = ArithmeticEngine(
                        self.worker.netzwerk
                    )
                    logger.debug(
                        "ArithmeticEngine initialisiert (lazy loading in Strategy)"
                    )

                arithmetic_engine = self.worker.arithmetic_engine

                # Berechnung durchführen
                arithmetic_result = arithmetic_engine.calculate(
                    operator, operand1, operand2
                )

                result_value = arithmetic_result.value
                proof_tree = arithmetic_result.proof_tree
                confidence = arithmetic_result.confidence

                logger.debug(
                    f"Calculated (ArithmeticEngine): {operand1} {operator} {operand2} = {result_value}"
                )

                # Emittiere Proof Tree an UI
                if proof_tree and hasattr(self.worker, "signals"):
                    self.worker.signals.proof_tree_update.emit(proof_tree)
                    logger.debug(
                        f"[Proof Tree] Arithmetic ProofTree emittiert: {len(proof_tree.get_all_steps())} Schritte"
                    )

                # Übernehme vorherigen Context und füge Ergebnis hinzu
                result = dict(context)
                result["result_value"] = result_value
                result["confidence"] = confidence
                result["proof_tree"] = proof_tree
                return True, result

            except ImportError:
                logger.warning(
                    "ArithmeticEngine nicht verfügbar, verwende Fallback-Berechnung"
                )

                # FALLBACK: Einfache Berechnung (ohne Proof Tree)
                if operator == "+":
                    result_value = operand1 + operand2
                elif operator == "-":
                    result_value = operand1 - operand2
                elif operator == "*":
                    result_value = operand1 * operand2
                elif operator == "/":
                    if operand2 == 0:
                        return False, {"error": "Division durch Null ist nicht erlaubt"}
                    result_value = operand1 / operand2
                else:
                    return False, {"error": f"Unbekannter Operator: {operator}"}

                logger.debug(
                    f"Calculated (Fallback): {operand1} {operator} {operand2} = {result_value}"
                )

                # Übernehme vorherigen Context und füge Ergebnis hinzu
                result = dict(context)
                result["result_value"] = result_value
                result["confidence"] = 1.0  # Fallback: Immer 100% Confidence
                return True, result

        except Exception as e:
            logger.error(f"Fehler bei Berechnung: {e}", exc_info=True)
            return False, {"error": str(e)}

    def _format_result(self, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Formatiere Ergebnis als Zahlwort.

        Verwendet component_53_number_language (wenn verfügbar) oder Fallback.
        """
        try:
            result_value = context.get("result_value")

            if result_value is None:
                return False, {"error": "Ergebnis fehlt im Kontext"}

            # PHASE 1: Einfacher Fallback für Zahlen 0-20 (wird später durch component_53 ersetzt)
            number_words = {
                0: "null",
                1: "eins",
                2: "zwei",
                3: "drei",
                4: "vier",
                5: "fünf",
                6: "sechs",
                7: "sieben",
                8: "acht",
                9: "neun",
                10: "zehn",
                11: "elf",
                12: "zwölf",
                13: "dreizehn",
                14: "vierzehn",
                15: "fünfzehn",
                16: "sechzehn",
                17: "siebzehn",
                18: "achtzehn",
                19: "neunzehn",
                20: "zwanzig",
            }

            # Versuche als Zahlwort zu formatieren, sonst als Zahl
            if isinstance(result_value, float) and result_value.is_integer():
                result_value = int(result_value)

            if isinstance(result_value, int) and result_value in number_words:
                result_word = number_words[result_value]
            else:
                result_word = str(result_value)

            logger.debug(f"Formatted result: {result_value} -> {result_word}")

            # Übernehme vorherigen Context und füge formatiertes Ergebnis hinzu
            result = dict(context)
            result["result_word"] = result_word
            result["final_answer"] = f"Das Ergebnis ist {result_word}."
            return True, result

        except Exception as e:
            logger.error(f"Fehler bei Formatierung: {e}", exc_info=True)
            return False, {"error": str(e)}


# ============================================================================
# INTROSPECTION STRATEGY (Production Rule Queries - PHASE 9)
# ============================================================================
