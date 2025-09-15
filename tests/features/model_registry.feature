Feature: Model Registry System
  As a Vision AI developer
  I want a centralized model registry
  So that I can easily manage and discover available models

  Background:
    Given the model registry system is initialized
    And model registration decorators are available

  Scenario: Model Registration
    Given I have a custom vision model class
    When I register the model with the registry
    Then the model should be stored in the registry
    And it should be accessible by its registered name
    And it should include all model metadata

  Scenario: Model Discovery by Name
    Given models are registered in the registry
    When I search for a model by name "resnet"
    Then the registry should return the correct model class
    And the model should have proper metadata
    And the model should be instantiable

  Scenario: Model Discovery by Task Type
    Given models are registered for different task types
    When I search for models by task type "classification"
    Then the registry should return all classification models
    And each model should support the specified task type
    And the results should be properly formatted

  Scenario: Model Aliases Support
    Given a model is registered with aliases
    When I search for the model using an alias
    Then the registry should return the correct model
    And the alias should resolve to the main model name
    And all aliases should work consistently

  Scenario: Model Registry Validation
    Given I attempt to register an invalid model
    When the registration process runs
    Then the registry should validate the model class
    And it should reject models without required methods
    And appropriate error messages should be provided

  Scenario: Model Registry Listing
    Given multiple models are registered
    When I request a list of all available models
    Then the registry should return a complete list
    And each entry should include model metadata
    And the list should be properly formatted

  Scenario: Model Registry Clear and Reset
    Given models are registered in the registry
    When I clear the registry
    Then all registered models should be removed
    And the registry should be empty
    And subsequent registrations should work normally