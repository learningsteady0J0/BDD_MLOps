Feature: Integration Workflow
  As a Vision AI researcher
  I want end-to-end training workflows
  So that I can seamlessly train models from configuration to deployment

  Background:
    Given the complete framework is available
    And all components are properly integrated

  Scenario: Complete Training Pipeline
    Given a valid experiment configuration
    When I start a complete training workflow
    Then the data module should be initialized
    And the model should be created and configured
    And the trainer should be set up with proper callbacks
    And training should execute successfully
    And results should be logged and saved

  Scenario: CLI Training Execution
    Given the main.py script is available
    When I execute training via command line
    Then the configuration should be loaded from files
    And all components should be initialized properly
    And training should proceed without errors
    And logs and checkpoints should be saved correctly

  Scenario: GUI Training Execution
    Given the GUI application is running
    When I configure and start training through the GUI
    Then the training should execute in a background thread
    And progress should be displayed in real-time
    And the GUI should remain responsive
    And users should be able to monitor and control training

  Scenario: Model Checkpointing and Recovery
    Given a training session is in progress
    When training is interrupted and resumed
    Then the model should resume from the latest checkpoint
    And training state should be properly restored
    And metrics continuity should be maintained

  Scenario: Experiment Tracking Integration
    Given MLflow is configured for experiment tracking
    When I run a training experiment
    Then experiment metadata should be logged
    And model metrics should be tracked over time
    And model artifacts should be saved
    And experiments should be comparable and reproducible

  Scenario: Multi-GPU Training Support
    Given multiple GPUs are available
    When I configure training for multi-GPU setup
    Then the framework should utilize all available GPUs
    And training should be properly distributed
    And results should be consistent with single-GPU training

  Scenario: Hyperparameter Optimization Integration
    Given a hyperparameter search configuration
    When I run hyperparameter optimization
    Then multiple experiments should be executed
    And the best parameters should be identified
    And results should be properly logged and compared

  Scenario: Model Evaluation and Testing
    Given a trained model is available
    When I run evaluation on test data
    Then comprehensive metrics should be computed
    And results should be properly formatted and saved
    And evaluation should be reproducible

  Scenario: Error Handling and Recovery
    Given various failure scenarios
    When errors occur during training
    Then appropriate error messages should be displayed
    And the system should handle errors gracefully
    And recovery options should be available where possible

  Scenario: Resource Monitoring and Optimization
    Given training is in progress
    When I monitor system resources
    Then CPU and GPU utilization should be tracked
    And memory usage should be monitored
    And performance bottlenecks should be identified
    And resource usage should be optimized