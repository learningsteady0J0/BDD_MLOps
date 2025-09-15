Feature: Configuration Management System
  As a Vision AI developer
  I want flexible configuration management
  So that I can easily manage experimental setups

  Background:
    Given the Hydra configuration system is available
    And configuration schemas are defined

  Scenario: Configuration Loading
    Given a valid configuration file exists
    When I load the configuration
    Then it should parse all configuration sections
    And it should validate against the schema
    And it should provide proper error messages for invalid configs

  Scenario: Configuration Overrides
    Given a base configuration is loaded
    When I override specific parameters
    Then the new values should take precedence
    And the configuration should remain valid
    And other parameters should remain unchanged

  Scenario: Configuration Composition
    Given multiple configuration files
    When I compose them together
    Then the final configuration should merge properly
    And conflicting values should be resolved correctly
    And the composition order should be respected

  Scenario: Environment-Specific Configurations
    Given configurations for different environments
    When I select a specific environment
    Then the appropriate configuration should be loaded
    And environment-specific overrides should be applied
    And the configuration should be complete and valid

  Scenario: Configuration Validation
    Given a configuration with invalid values
    When I attempt to validate it
    Then validation errors should be detected
    And helpful error messages should be provided
    And the validation should be comprehensive

  Scenario: Dynamic Configuration Updates
    Given a running application with loaded configuration
    When I update configuration parameters
    Then the changes should be applied safely
    And dependent components should be notified
    And the system should remain stable

  Scenario: Configuration Templates and Presets
    Given predefined configuration templates
    When I select a template for a specific task
    Then the appropriate configuration should be generated
    And all required parameters should have default values
    And the template should be customizable