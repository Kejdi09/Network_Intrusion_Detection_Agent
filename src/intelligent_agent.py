"""
Intelligent Agent System for Network Intrusion Detection
Implements: Rule-Based Reasoning, Planning, and Adaptive Learning
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple

class ThreatLevel(Enum):
    """Threat classification levels"""
    SAFE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ActionType(Enum):
    """Types of actions the agent can take"""
    ALLOW = "allow"
    LOG = "log"
    RATE_LIMIT = "rate_limit"
    ALERT = "alert"
    ISOLATE = "isolate"
    BLOCK = "block"
    ESCALATE = "escalate"

@dataclass
class Decision:
    """Represents an intelligent decision with reasoning"""
    action: ActionType
    threat_level: ThreatLevel
    confidence: float
    rules_triggered: List[str]
    remediation_plan: List[str]
    explanation: str

class ExpertSystem:
    """Rule-based expert system for threat assessment"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.decisions_log = []
        self.feedback_scores = {}
    
    def _initialize_rules(self) -> Dict:
        """Initialize expert system rules"""
        return {
            # High confidence rules
            "critical_malicious": {
                "conditions": {
                    "malicious_prob": lambda x: x > 0.95,
                    "confidence": lambda x: x > 0.90,
                },
                "action": ActionType.BLOCK,
                "threat_level": ThreatLevel.CRITICAL,
                "priority": 1
            },
            
            "high_confidence_malicious": {
                "conditions": {
                    "malicious_prob": lambda x: x > 0.80,
                    "confidence": lambda x: x > 0.75,
                },
                "action": ActionType.ISOLATE,
                "threat_level": ThreatLevel.HIGH,
                "priority": 2
            },
            
            "anomalous_malicious": {
                "conditions": {
                    "malicious_prob": lambda x: x > 0.60,
                    "anomaly_score": lambda x: x > 0.75,
                },
                "action": ActionType.ALERT,
                "threat_level": ThreatLevel.HIGH,
                "priority": 3
            },
            
            "moderate_malicious": {
                "conditions": {
                    "malicious_prob": lambda x: 0.50 < x <= 0.80,
                    "confidence": lambda x: x > 0.60,
                },
                "action": ActionType.ALERT,
                "threat_level": ThreatLevel.MEDIUM,
                "priority": 4
            },
            
            "suspicious_pattern": {
                "conditions": {
                    "malicious_prob": lambda x: x > 0.40,
                    "anomaly_score": lambda x: x > 0.60,
                },
                "action": ActionType.LOG,
                "threat_level": ThreatLevel.MEDIUM,
                "priority": 5
            },
            
            "high_anomaly_benign": {
                "conditions": {
                    "benign_prob": lambda x: x > 0.50,
                    "anomaly_score": lambda x: x > 0.70,
                },
                "action": ActionType.ALERT,
                "threat_level": ThreatLevel.MEDIUM,
                "priority": 6
            },
            
            "confident_benign": {
                "conditions": {
                    "benign_prob": lambda x: x > 0.90,
                    "anomaly_score": lambda x: x < 0.40,
                },
                "action": ActionType.ALLOW,
                "threat_level": ThreatLevel.SAFE,
                "priority": 7
            },
            
            "benign_normal": {
                "conditions": {
                    "benign_prob": lambda x: x > 0.60,
                    "anomaly_score": lambda x: x < 0.30,
                },
                "action": ActionType.ALLOW,
                "threat_level": ThreatLevel.SAFE,
                "priority": 8
            },
        }
    
    def evaluate(self, 
                malicious_prob: float, 
                benign_prob: float,
                anomaly_score: float,
                attack_type: str = None,
                attack_confidence: float = 0.0) -> Decision:
        """
        Evaluate threat and determine best action using expert rules
        
        Args:
            malicious_prob: Probability of malicious traffic (0-1)
            benign_prob: Probability of benign traffic (0-1)
            anomaly_score: Anomaly score (0-1)
            attack_type: Type of attack if identified
            attack_confidence: Confidence in attack type identification
        
        Returns:
            Decision object with action and reasoning
        """
        
        triggered_rules = []
        confidence = max(malicious_prob, benign_prob)
        
        # Check rules in priority order
        for rule_name, rule in sorted(
            self.rules.items(),
            key=lambda x: x[1].get("priority", 999)
        ):
            if self._check_rule(rule, malicious_prob, benign_prob, anomaly_score):
                triggered_rules.append(rule_name)
                
                # Use first matching rule (highest priority)
                action = rule["action"]
                threat_level = rule["threat_level"]
                
                remediation = self._generate_remediation(
                    action, attack_type, threat_level, attack_confidence
                )
                
                explanation = self._generate_explanation(
                    rule_name, triggered_rules, malicious_prob, 
                    benign_prob, anomaly_score, attack_type
                )
                
                decision = Decision(
                    action=action,
                    threat_level=threat_level,
                    confidence=confidence,
                    rules_triggered=triggered_rules,
                    remediation_plan=remediation,
                    explanation=explanation
                )
                
                self.decisions_log.append(decision)
                return decision
        
        # Fallback rule (should not reach here)
        return Decision(
            action=ActionType.LOG,
            threat_level=ThreatLevel.LOW,
            confidence=confidence,
            rules_triggered=["default_fallback"],
            remediation_plan=["Continue monitoring"],
            explanation="No specific rule matched; default to logging."
        )
    
    def _check_rule(self, rule: Dict, malicious_prob: float, 
                   benign_prob: float, anomaly_score: float) -> bool:
        """Check if all conditions in a rule are satisfied"""
        conditions = rule.get("conditions", {})
        
        params = {
            "malicious_prob": malicious_prob,
            "benign_prob": benign_prob,
            "anomaly_score": anomaly_score,
        }
        
        for param_name, condition_func in conditions.items():
            if param_name in params:
                if not condition_func(params[param_name]):
                    return False
        
        return True
    
    def _generate_remediation(self, action: ActionType, attack_type: str,
                             threat_level: ThreatLevel, confidence: float) -> List[str]:
        """Generate remediation/action plan based on threat"""
        
        plans = {
            ActionType.ALLOW: [
                "✓ Traffic permitted",
                "Continue normal processing",
                "Log for statistics"
            ],
            
            ActionType.LOG: [
                "📝 Log packet details",
                "Monitor for patterns",
                "Alert if pattern repeats"
            ],
            
            ActionType.RATE_LIMIT: [
                "⚠️ Apply rate limiting",
                "Monitor traffic volume",
                "Adjust limits if needed"
            ],
            
            ActionType.ALERT: [
                "🚨 Generate security alert",
                "Notify security team",
                "Review packet characteristics"
            ],
            
            ActionType.ISOLATE: [
                "🔒 Isolate source/destination",
                "Preserve connection logs",
                "Initiate investigation",
                "Review other connections from source"
            ],
            
            ActionType.BLOCK: [
                "⛔ Block immediately",
                "Log all packet details",
                "Add to blacklist",
                "Notify SOC team",
                "Prepare incident report"
            ],
            
            ActionType.ESCALATE: [
                "📞 Escalate to security team",
                "Preserve evidence",
                "Request human review",
                "Await expert decision"
            ]
        }
        
        base_plan = plans.get(action, ["Unknown action"])
        
        # Add attack-specific recommendations
        if attack_type:
            specific_recommendations = self._get_attack_specific_actions(attack_type)
            base_plan.extend(specific_recommendations)
        
        # Add threat level considerations
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            base_plan.extend([
                "🔴 HIGH PRIORITY RESPONSE",
                "Prepare emergency incident response",
                "Document all actions taken"
            ])
        
        return base_plan
    
    def _get_attack_specific_actions(self, attack_type: str) -> List[str]:
        """Get specific actions for attack type"""
        
        attack_actions = {
            "DDoS": [
                "Enable DDoS protection",
                "Rate limit traffic",
                "Activate WAF rules",
                "Redirect to scrubbing center if available"
            ],
            "DDoS-UDP": [
                "Filter UDP traffic",
                "Rate limit source",
                "Enable reflection attack protection"
            ],
            "DDoS-TCP": [
                "Filter TCP traffic",
                "Implement connection limits",
                "Enable SYN flood protection"
            ],
            "Botnet": [
                "Isolate compromised device",
                "Clean malware",
                "Change all credentials",
                "Update antivirus signatures"
            ],
            "Backdoor": [
                "Isolate system immediately",
                "Preserve forensic evidence",
                "Initiate incident response",
                "Review access logs",
                "Change all credentials"
            ],
            "Trojan": [
                "Remove malicious software",
                "Isolate system",
                "Scan for data exfiltration",
                "Update security software"
            ],
            "Worm": [
                "Apply security patches",
                "Isolate network segment",
                "Update signature definitions",
                "Monitor for propagation"
            ],
            "Exploit": [
                "Apply security patch immediately",
                "Isolate vulnerable system",
                "Review system logs for compromise",
                "Activate intrusion prevention"
            ],
            "Reconnaissance": [
                "Block scanning source",
                "Monitor for actual attack",
                "Harden exposed services",
                "Implement network segmentation"
            ],
        }
        
        return attack_actions.get(attack_type, ["Monitor closely for follow-up attacks"])
    
    def _generate_explanation(self, rule_name: str, triggered_rules: List[str],
                             malicious_prob: float, benign_prob: float,
                             anomaly_score: float, attack_type: str) -> str:
        """Generate human-readable explanation of decision"""
        
        explanations = {
            "critical_malicious": 
                f"✗ CRITICAL THREAT DETECTED\n"
                f"Malicious probability is extremely high ({malicious_prob:.1%}). "
                f"System has very high confidence in malicious classification. "
                f"Immediate blocking recommended.",
            
            "high_confidence_malicious":
                f"✗ HIGH-CONFIDENCE MALICIOUS TRAFFIC\n"
                f"Malicious probability {malicious_prob:.1%} combined with high confidence ({benign_prob:.1%}). "
                f"Source should be isolated for investigation.",
            
            "anomalous_malicious":
                f"⚠️ SUSPICIOUS ANOMALOUS PATTERN\n"
                f"Malicious probability {malicious_prob:.1%} with highly anomalous behavior ({anomaly_score:.1%}). "
                f"Pattern deviation + malicious indicators warrant investigation.",
            
            "moderate_malicious":
                f"⚠️ MODERATE THREAT DETECTED\n"
                f"Malicious probability {malicious_prob:.1%} suggests possible attack. "
                f"Close monitoring and potential isolation recommended.",
            
            "suspicious_pattern":
                f"📊 SUSPICIOUS PATTERN DETECTED\n"
                f"Unusual behavior detected (anomaly score: {anomaly_score:.1%}) with moderate malicious indicators. "
                f"Recommend logging and monitoring.",
            
            "high_anomaly_benign":
                f"⚠️ ANOMALOUS BENIGN TRAFFIC\n"
                f"Traffic appears benign ({benign_prob:.1%}) but shows unusual patterns ({anomaly_score:.1%}). "
                f"May indicate legitimate but unusual activity or spoofed benign traffic.",
            
            "confident_benign":
                f"✓ LEGITIMATE TRAFFIC CONFIRMED\n"
                f"Very high confidence in benign classification ({benign_prob:.1%}) with normal behavior. "
                f"Traffic is safe to allow.",
            
            "benign_normal":
                f"✓ NORMAL BENIGN TRAFFIC\n"
                f"Traffic appears benign ({benign_prob:.1%}) with normal patterns. "
                f"No threats detected.",
        }
        
        explanation = explanations.get(rule_name, "Unable to determine explanation.")
        
        if attack_type:
            explanation += f"\n\nDetected Attack Type: {attack_type}"
        
        return explanation
    
    def get_decision_stats(self) -> Dict:
        """Get statistics about decisions made"""
        if not self.decisions_log:
            return {"total_decisions": 0}
        
        from collections import Counter
        
        actions = [d.action.value for d in self.decisions_log]
        threats = [d.threat_level.name for d in self.decisions_log]
        
        return {
            "total_decisions": len(self.decisions_log),
            "actions": dict(Counter(actions)),
            "threat_levels": dict(Counter(threats)),
            "avg_confidence": np.mean([d.confidence for d in self.decisions_log]),
            "last_decision": self.decisions_log[-1] if self.decisions_log else None
        }

class PlanningAgent:
    """Planning agent for generating action sequences"""
    
    def __init__(self):
        self.action_history = []
        self.learned_plans = {}
    
    def generate_response_plan(self, threat_level: ThreatLevel, 
                              attack_type: str, confidence: float) -> List[Tuple[str, str]]:
        """
        Generate sequential response plan (Action, Reasoning)
        Implements simple planning algorithm
        """
        plan = []
        
        # Phase 1: Immediate response
        plan.append(("DETECT", f"Threat detected: {attack_type} at {confidence:.1%} confidence"))
        
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            plan.append(("ISOLATE", "Initiate source isolation"))
            plan.append(("ALERT", "Send emergency alert to SOC"))
        elif threat_level == ThreatLevel.MEDIUM:
            plan.append(("LOG", "Log incident details"))
            plan.append(("ALERT", "Send alert to security team"))
        else:
            plan.append(("LOG", "Log for monitoring"))
        
        # Phase 2: Investigation
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            plan.append(("INVESTIGATE", "Analyze packet signatures"))
            plan.append(("FORENSICS", "Preserve evidence for incident response"))
            plan.append(("CORRELATE", "Check for related incidents"))
        
        # Phase 3: Remediation
        if attack_type == "DDoS":
            plan.append(("ENABLE_PROTECTION", "Activate DDoS mitigation"))
            plan.append(("RATE_LIMIT", "Apply rate limiting rules"))
        elif attack_type in ["Botnet", "Backdoor", "Trojan"]:
            plan.append(("ISOLATE_SYSTEM", "Isolate compromised device"))
            plan.append(("CLEAN", "Initiate system remediation"))
        elif attack_type == "Worm":
            plan.append(("PATCH", "Apply security updates"))
            plan.append(("SCAN", "Scan for propagation"))
        
        # Phase 4: Follow-up
        plan.append(("MONITOR", "Continuous monitoring for recurrence"))
        plan.append(("DOCUMENT", "Create incident report"))
        
        self.action_history.extend(plan)
        return plan

class AdaptivelearningSystem:
    """Adaptive learning system to improve detection over time"""
    
    def __init__(self):
        self.feedback_data = []
        self.threshold_adjustments = {}
        self.accuracy_metrics = {}
    
    def record_feedback(self, prediction: str, actual: str, confidence: float):
        """Record prediction feedback for learning"""
        self.feedback_data.append({
            "prediction": prediction,
            "actual": actual,
            "confidence": confidence,
            "correct": prediction == actual
        })
        
        self._update_accuracy_metrics()
    
    def _update_accuracy_metrics(self):
        """Update accuracy metrics from feedback"""
        if not self.feedback_data:
            return
        
        total = len(self.feedback_data)
        correct = sum(1 for f in self.feedback_data if f["correct"])
        
        self.accuracy_metrics = {
            "accuracy": correct / total,
            "total_samples": total,
            "correct_predictions": correct,
            "confidence_avg": np.mean([f["confidence"] for f in self.feedback_data])
        }
    
    def suggest_threshold_adjustment(self) -> Dict:
        """Suggest threshold adjustments based on feedback"""
        if len(self.feedback_data) < 10:
            return {"status": "Insufficient data for adjustment"}
        
        # Find patterns where model was wrong
        wrong_predictions = [f for f in self.feedback_data if not f["correct"]]
        
        if not wrong_predictions:
            return {"status": "Perfect accuracy - no adjustments needed"}
        
        # Suggest adjustments
        false_positives = [f for f in wrong_predictions if f["prediction"] == "malicious"]
        false_negatives = [f for f in wrong_predictions if f["prediction"] == "benign"]
        
        suggestions = {}
        
        if false_positives:
            avg_confidence = np.mean([f["confidence"] for f in false_positives])
            suggestions["raise_malicious_threshold"] = f"Increase from current level, many FP at {avg_confidence:.1%}"
        
        if false_negatives:
            avg_confidence = np.mean([f["confidence"] for f in false_negatives])
            suggestions["lower_benign_threshold"] = f"Decrease threshold, missing attacks at {avg_confidence:.1%}"
        
        return suggestions
    
    def get_performance_report(self) -> Dict:
        """Generate performance report"""
        return {
            **self.accuracy_metrics,
            "feedback_samples": len(self.feedback_data),
            "suggested_adjustments": self.suggest_threshold_adjustment()
        }
