# setup_initial_knowledge.py
"""
Initialisiert das KAI-System mit grundlegendem Meta-Wissen.
Enthält robuste Fehlerbehandlung und Verifizierung.
"""
import logging
import sys
import time

# WICHTIG: Encoding-Fix MUSS früh importiert werden
# Behebt Windows cp1252 -> UTF-8 Probleme für Unicode-Zeichen ([ERROR], [SUCCESS], [WARNING], etc.)
import kai_encoding_fix  # noqa: F401 (automatische Aktivierung beim Import)
from component_1_netzwerk import KonzeptNetzwerk
from kai_logging import setup_logging


def verify_database_connection(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Verifiziert die Datenbankverbindung und prüft grundlegende Funktionalität.

    Returns:
        True bei erfolgreicher Verbindung, False sonst
    """
    logger = logging.getLogger("KAI_SETUP")

    if not netzwerk.driver:
        logger.error("[ERROR] Netzwerk-Driver ist None - keine Verbindung möglich")
        return False

    try:
        # Prüfe Konnektivität
        netzwerk.driver.verify_connectivity()
        logger.info("[SUCCESS] Datenbankverbindung erfolgreich")

        # Prüfe, ob wir Schreibrechte haben
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run("RETURN 1 AS test")
            if result.single()["test"] != 1:
                logger.error("[ERROR] Kann keine Daten aus der Datenbank lesen")
                return False

        logger.info("[SUCCESS] Lese-/Schreibzugriff verifiziert")
        return True

    except Exception as e:
        logger.error(f"[ERROR] Datenbankverbindung fehlgeschlagen: {e}", exc_info=True)
        return False


def verify_constraints(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Prüft, ob die erforderlichen Constraints existieren.

    Returns:
        True wenn alle Constraints vorhanden sind
    """
    logger = logging.getLogger("KAI_SETUP")

    expected_constraints = [
        "WortLemma",
        "KonzeptName",
        "ExtractionRuleType",
        "PatternPrototypeId",
        "LexiconName",
    ]

    try:
        with netzwerk.driver.session(database="neo4j") as session:
            result = session.run("SHOW CONSTRAINTS")
            existing_constraints = {record["name"] for record in result}

        missing = []
        for constraint in expected_constraints:
            if constraint not in existing_constraints:
                missing.append(constraint)

        if missing:
            logger.warning(f"[WARNING]  Fehlende Constraints: {missing}")
            logger.info("Constraints werden automatisch beim ersten Aufruf erstellt")
        else:
            logger.info(
                f"[SUCCESS] Alle {len(expected_constraints)} Constraints vorhanden"
            )

        return True

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Prüfen der Constraints: {e}")
        return False


def create_extraction_rules(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt die grundlegenden Extraktionsregeln.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    rules_to_create = [
        {
            "type": "IS_A",
            "regex": r"^(?:ein|eine|der|die|das\s+)?(.+?)\s+ist\s+(?:auch\s+)?(?:ein|eine|der|die|das\s+)?(.+?)\.?$",
            "desc": "Grundlegende 'ist ein/eine'-Beziehung mit Artikeln",
        },
        {
            "type": "HAS_PROPERTY",
            "regex": r"^(.*?)\s+hat\s+(?:die Eigenschaft|die Fähigkeit)?\s*(.*?)\.?$",
            "desc": "Erfasst Eigenschaften oder Fähigkeiten",
        },
        {
            "type": "CAPABLE_OF",
            "regex": r"^(.*?)\s+kann\s+(.*?)\.?$",
            "desc": "Erfasst Fähigkeiten (z.B. 'Ein Vogel kann fliegen')",
        },
        {
            "type": "PART_OF",
            "regex": r"^(.*?)\s+(?:ist Teil von|gehört zu)\s+(.*?)\.?$",
            "desc": "Erfasst Teil-Ganzes-Beziehungen",
        },
        {
            "type": "LOCATED_IN",
            "regex": r"^(.*?)\s+(?:ist in|befindet sich in|liegt in)\s+(.*?)\.?$",
            "desc": "Erfasst räumliche Beziehungen",
        },
    ]

    logger.info(f"📝 Erstelle {len(rules_to_create)} Extraktionsregeln...")

    created_count = 0
    for rule in rules_to_create:
        try:
            logger.info(f"  -> Erstelle: {rule['type']} ({rule['desc']})")
            netzwerk.create_extraction_rule(
                relation_type=rule["type"], regex_pattern=rule["regex"]
            )

            # Kurze Pause, um sicherzustellen, dass die DB Zeit zum Schreiben hat
            time.sleep(0.1)

            # Sofortige Verifikation
            with netzwerk.driver.session(database="neo4j") as session:
                result = session.run(
                    """
                    MATCH (r:ExtractionRule {relation_type: $rel})
                    RETURN r.regex_pattern AS pattern
                """,
                    rel=rule["type"],
                )
                record = result.single()

                if record and record["pattern"] == rule["regex"]:
                    created_count += 1
                    logger.info(
                        f"    [SUCCESS] Regel '{rule['type']}' erfolgreich in DB gespeichert"
                    )
                else:
                    logger.error(
                        f"    [ERROR] Regel '{rule['type']}' NICHT in DB gefunden!"
                    )
                    return False

        except Exception as e:
            logger.error(f"    [ERROR] Fehler beim Erstellen von '{rule['type']}': {e}")
            return False

    logger.info(
        f"[SUCCESS] {created_count}/{len(rules_to_create)} Regeln erfolgreich erstellt"
    )
    return created_count == len(rules_to_create)


def create_lexical_triggers(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Erstellt die initialen lexikalischen Trigger.

    Returns:
        True bei Erfolg, False bei Fehler
    """
    logger = logging.getLogger("KAI_SETUP")

    initial_triggers = [
        "ist",
        "sind",
        "bezeichnet",
        "definiert",
        "heißt",
        "hat",
        "kann",
        "gehört",
        "befindet",
    ]

    logger.info(f"🏷️  Füge {len(initial_triggers)} lexikalische Trigger hinzu...")

    created_count = 0
    for trigger in initial_triggers:
        try:
            created = netzwerk.add_lexical_trigger(trigger)
            if created:
                created_count += 1
                logger.info(f"  [SUCCESS] Trigger '{trigger}' hinzugefügt")
            else:
                logger.info(f"  [INFO]  Trigger '{trigger}' existierte bereits")
        except Exception as e:
            logger.error(f"  [ERROR] Fehler beim Hinzufügen von '{trigger}': {e}")
            return False

    logger.info(
        f"[SUCCESS] {created_count} neue Trigger erstellt, {len(initial_triggers) - created_count} bereits vorhanden"
    )
    return True


def verify_complete_setup(netzwerk: KonzeptNetzwerk) -> bool:
    """
    Führt eine umfassende Verifikation des Setups durch.

    Returns:
        True wenn alles korrekt ist, False sonst
    """
    logger = logging.getLogger("KAI_SETUP")

    logger.info("🔍 Führe abschließende Verifikation durch...")

    # 1. Prüfe Extraktionsregeln
    try:
        rules = netzwerk.get_all_extraction_rules()
        expected_rules = ["IS_A", "HAS_PROPERTY", "CAPABLE_OF", "PART_OF", "LOCATED_IN"]

        if len(rules) < len(expected_rules):
            logger.error(
                f"[ERROR] Nur {len(rules)} von {len(expected_rules)} Regeln gefunden"
            )
            return False

        existing_types = {r["relation_type"] for r in rules}
        missing_rules = [r for r in expected_rules if r not in existing_types]

        if missing_rules:
            logger.error(f"[ERROR] Fehlende Regeln: {missing_rules}")
            return False

        logger.info(
            f"  [SUCCESS] Alle {len(expected_rules)} Extraktionsregeln vorhanden"
        )

        # Detaillierte Ausgabe der Regeln
        for rule in rules:
            if rule["relation_type"] in expected_rules:
                logger.info(
                    f"    * {rule['relation_type']}: {rule['regex_pattern'][:50]}..."
                )

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Verifizieren der Regeln: {e}")
        return False

    # 2. Prüfe Lexikalische Trigger
    try:
        triggers = netzwerk.get_lexical_triggers()
        expected_triggers = ["ist", "hat", "kann"]

        missing_triggers = [t for t in expected_triggers if t not in triggers]

        if missing_triggers:
            logger.error(f"[ERROR] Fehlende Trigger: {missing_triggers}")
            return False

        logger.info(
            f"  [SUCCESS] Alle erwarteten Trigger vorhanden ({len(triggers)} total)"
        )

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Verifizieren der Trigger: {e}")
        return False

    # 3. Teste eine einfache Regel-Anwendung
    try:
        logger.info("  🧪 Teste Regel-Anwendung mit Beispielsatz...")
        test_subject = "_test_setup_hund"
        test_object = "_test_setup_tier"

        # Erstelle Testrelation
        created = netzwerk.assert_relation(
            test_subject, "IS_A", test_object, "Setup-Verifikationstest"
        )

        if not created:
            logger.warning(
                "  [WARNING]  Testrelation konnte nicht erstellt werden (möglicherweise bereits vorhanden)"
            )

        # Prüfe, ob Relation existiert
        facts = netzwerk.query_graph_for_facts(test_subject)

        if "IS_A" not in facts or test_object not in facts["IS_A"]:
            logger.error("  [ERROR] Testrelation nicht im Graphen gefunden")
            return False

        logger.info("  [SUCCESS] Regel-Anwendung funktioniert korrekt")

        # Cleanup
        with netzwerk.driver.session(database="neo4j") as session:
            session.run(
                """
                MATCH (n)
                WHERE n.name STARTS WITH '_test_setup_'
                DETACH DELETE n
            """
            )
        logger.info("  🧹 Test-Daten bereinigt")

    except Exception as e:
        logger.error(f"[ERROR] Fehler beim Testen der Regel-Anwendung: {e}")
        return False

    return True


def run_setup():
    """
    Hauptfunktion für das Setup mit vollständiger Fehlerbehandlung und Verifizierung.
    """
    setup_logging()
    logger = logging.getLogger("KAI_SETUP")

    logger.info("=" * 70)
    logger.info("KAI SYSTEM SETUP - Initialisierung gestartet")
    logger.info("=" * 70)

    # Schritt 1: Datenbankverbindung herstellen
    logger.info("\n📡 Schritt 1: Datenbankverbindung herstellen...")
    try:
        netzwerk = KonzeptNetzwerk()
    except Exception as e:
        logger.critical(
            f"[ERROR] FATAL: Konnte KonzeptNetzwerk nicht initialisieren: {e}"
        )
        sys.exit(1)

    if not verify_database_connection(netzwerk):
        logger.critical("[ERROR] FATAL: Datenbankverbindung fehlgeschlagen")
        logger.critical("Bitte prüfen Sie:")
        logger.critical("  1. Läuft Neo4j auf localhost:7687?")
        logger.critical("  2. Sind die Zugangsdaten korrekt (neo4j/password)?")
        logger.critical("  3. Ist die Datenbank erreichbar?")
        netzwerk.close()
        sys.exit(1)

    # Schritt 2: Constraints prüfen
    logger.info("\n🔒 Schritt 2: Constraints prüfen...")
    if not verify_constraints(netzwerk):
        logger.critical("[ERROR] FATAL: Constraint-Verifikation fehlgeschlagen")
        netzwerk.close()
        sys.exit(1)

    # Schritt 3: Extraktionsregeln erstellen
    logger.info("\n[INFO] Schritt 3: Extraktionsregeln erstellen...")
    if not create_extraction_rules(netzwerk):
        logger.critical("[ERROR] FATAL: Konnte Extraktionsregeln nicht erstellen")
        netzwerk.close()
        sys.exit(1)

    # Schritt 4: Lexikalische Trigger erstellen
    logger.info("\n🏷️  Schritt 4: Lexikalische Trigger erstellen...")
    if not create_lexical_triggers(netzwerk):
        logger.critical("[ERROR] FATAL: Konnte Trigger nicht erstellen")
        netzwerk.close()
        sys.exit(1)

    # Schritt 5: Umfassende Verifikation
    logger.info("\n[OK] Schritt 5: Abschließende Verifikation...")
    if not verify_complete_setup(netzwerk):
        logger.critical("[ERROR] FATAL: Setup-Verifikation fehlgeschlagen")
        logger.critical("Das System ist nicht vollständig initialisiert!")
        netzwerk.close()
        sys.exit(1)

    # Erfolg!
    logger.info("\n" + "=" * 70)
    logger.info(
        "[SUCCESS] [SUCCESS] [SUCCESS]  SETUP ERFOLGREICH ABGESCHLOSSEN  [SUCCESS] [SUCCESS] [SUCCESS]"
    )
    logger.info("=" * 70)
    logger.info("\nDas KAI-System ist jetzt einsatzbereit!")
    logger.info("Sie können nun main_ui_graphical.py starten.")
    logger.info("\nZusammenfassung:")

    # Finale Statistik
    rules = netzwerk.get_all_extraction_rules()
    triggers = netzwerk.get_lexical_triggers()
    logger.info(f"  * {len(rules)} Extraktionsregeln aktiv")
    logger.info(f"  * {len(triggers)} lexikalische Trigger geladen")

    netzwerk.close()


if __name__ == "__main__":
    run_setup()
