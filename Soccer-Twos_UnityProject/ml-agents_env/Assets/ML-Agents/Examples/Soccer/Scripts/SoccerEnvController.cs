using System.Collections.Generic;
using Unity.MLAgents;
using UnityEngine;

public class SoccerEnvController : MonoBehaviour
{
    [System.Serializable]
    public class PlayerInfo
    {
        public AgentSoccer Agent;
        [HideInInspector]
        public Vector3 StartingPos;
        [HideInInspector]
        public Quaternion StartingRot;
        [HideInInspector]
        public Rigidbody Rb;
    }


    /// <summary>
    /// Max Academy steps before this platform resets
    /// </summary>
    /// <returns></returns>
    [Tooltip("Max Environment Steps")] public int MaxEnvironmentSteps = 25000;

    /// <summary>
    /// The area bounds.
    /// </summary>

    /// <summary>
    /// We will be changing the ground material based on success/failue
    /// </summary>

    public GameObject ball;
    [HideInInspector]
    public Rigidbody ballRb;
    Vector3 m_BallStartingPos;

    //List of Agents On Platform
    public List<PlayerInfo> AgentsList = new List<PlayerInfo>();

    private SoccerSettings m_SoccerSettings;


    private SimpleMultiAgentGroup m_BlueAgentGroup;
    private SimpleMultiAgentGroup m_PurpleAgentGroup;

    private int m_ResetTimer;

    private Transform blueGoalTransform; // Transform of the blue team's goal
    private Transform purpleGoalTransform; // Transform of the purple team's goal
    private float previousDistanceToBlueGoal = float.MaxValue;
    private float previousDistanceToPurpleGoal = float.MaxValue;
    private Team lastTeamToTouchBall;

    void Start()
    {

        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        // Initialize TeamManager
        m_BlueAgentGroup = new SimpleMultiAgentGroup();
        m_PurpleAgentGroup = new SimpleMultiAgentGroup();
        ballRb = ball.GetComponent<Rigidbody>();
        m_BallStartingPos = new Vector3(ball.transform.position.x, ball.transform.position.y, ball.transform.position.z);
        foreach (var item in AgentsList)
        {
            item.StartingPos = item.Agent.transform.position;
            item.StartingRot = item.Agent.transform.rotation;
            item.Rb = item.Agent.GetComponent<Rigidbody>();
            if (item.Agent.team == Team.Blue)
            {
                m_BlueAgentGroup.RegisterAgent(item.Agent);
            }
            else
            {
                m_PurpleAgentGroup.RegisterAgent(item.Agent);
            }
        }

        // Find and assign the goal transforms by tags
        GameObject blueGoal = GameObject.FindGameObjectWithTag("blueGoal");
        if (blueGoal != null)
        {
            blueGoalTransform = blueGoal.transform;
        }

        GameObject purpleGoal = GameObject.FindGameObjectWithTag("purpleGoal");
        if (purpleGoal != null)
        {
            purpleGoalTransform = purpleGoal.transform;
        }

        if (blueGoalTransform != null)
        {
            previousDistanceToBlueGoal = Vector3.Distance(ball.transform.position, blueGoalTransform.position);
        }

        if (purpleGoalTransform != null)
        {
            previousDistanceToPurpleGoal = Vector3.Distance(ball.transform.position, purpleGoalTransform.position);
        }

        ResetScene();
    }

    void FixedUpdate()
    {
        m_ResetTimer += 1;
        if (m_ResetTimer >= MaxEnvironmentSteps && MaxEnvironmentSteps > 0)
        {
            m_BlueAgentGroup.GroupEpisodeInterrupted();
            m_PurpleAgentGroup.GroupEpisodeInterrupted();
            ResetScene();
        }
        MoveBallTowardsOpponentAreaReward();
    }


    private void MoveBallTowardsOpponentAreaReward()
    {
        if (ball == null || blueGoalTransform == null || purpleGoalTransform == null) return;

        // Calculate the current distance to both goals
        float currentDistanceToBlueGoal = Vector3.Distance(ball.transform.position, blueGoalTransform.position);
        float currentDistanceToPurpleGoal = Vector3.Distance(ball.transform.position, purpleGoalTransform.position);

        // Check if the ball is closer to the blue goal compared to the last frame
        if (currentDistanceToBlueGoal < previousDistanceToBlueGoal)
        {
            // Reward purple team for moving the ball closer to the blue goal
            m_PurpleAgentGroup.AddGroupReward(0.002f);
            m_BlueAgentGroup.AddGroupReward(-0.001f);
        }

        // Check if the ball is closer to the purple goal compared to the last frame
        if (currentDistanceToPurpleGoal < previousDistanceToPurpleGoal)
        {
            // Reward blue team for moving the ball closer to the purple goal
            m_BlueAgentGroup.AddGroupReward(0.002f);
            m_PurpleAgentGroup.AddGroupReward(-0.001f);
        }

        // Update the previous distances for the next frame
        previousDistanceToBlueGoal = currentDistanceToBlueGoal;
        previousDistanceToPurpleGoal = currentDistanceToPurpleGoal;
    }


    public void ResetBall()
    {
        var randomPosX = Random.Range(-2.5f, 2.5f);
        var randomPosZ = Random.Range(-2.5f, 2.5f);

        ball.transform.position = m_BallStartingPos + new Vector3(randomPosX, 0f, randomPosZ);
        ballRb.velocity = Vector3.zero;
        ballRb.angularVelocity = Vector3.zero;

    }

    public void GoalTouched(Team scoredTeam, Transform goalHit)
    {
        float ownGoalPenalty = -2f; // Significant negative value for scoring an own goal

        // Check if it's an own goal
        bool isOwnGoal = IsOwnGoal(scoredTeam, goalHit);

        if (scoredTeam == Team.Blue)
        {
            if (isOwnGoal)
            {
                // Blue team scored an own goal
                m_BlueAgentGroup.AddGroupReward(ownGoalPenalty);
                m_PurpleAgentGroup.AddGroupReward(1); // Optionally reward the opposing team
            }
            else
            {
                // Blue team scored in the correct goal
                m_BlueAgentGroup.AddGroupReward(1 - (float)m_ResetTimer / MaxEnvironmentSteps);
                m_PurpleAgentGroup.AddGroupReward(-1);
            }
        }
        else // if scoredTeam == Team.Purple
        {
            if (isOwnGoal)
            {
                // Purple team scored an own goal
                m_PurpleAgentGroup.AddGroupReward(ownGoalPenalty);
                m_BlueAgentGroup.AddGroupReward(1); // Optionally reward the opposing team
            }
            else
            {
                // Purple team scored in the correct goal
                m_PurpleAgentGroup.AddGroupReward(1 - (float)m_ResetTimer / MaxEnvironmentSteps);
                m_BlueAgentGroup.AddGroupReward(-1);
            }
        }

        m_PurpleAgentGroup.EndGroupEpisode();
        m_BlueAgentGroup.EndGroupEpisode();
        ResetScene();
    }

    private bool IsOwnGoal(Team lastTeamToTouch, Transform goalHit)
    {
        if (lastTeamToTouch == Team.Blue && goalHit == blueGoalTransform)
        {
            // Blue team hit the ball into their own goal
            return true;
        }
        else if (lastTeamToTouch == Team.Purple && goalHit == purpleGoalTransform)
        {
            // Purple team hit the ball into their own goal
            return true;
        }

        // In other cases, it's not an own goal
        return false;
    }

    // Example of how to set the last team to touch the ball
    // This should be called whenever a team touches the ball
    public void SetLastTeamToTouchBall(Team team)
    {
        lastTeamToTouchBall = team;
    }


    public void ResetScene()
    {
        m_ResetTimer = 0;

        //Reset Agents
        foreach (var item in AgentsList)
        {
            var randomPosX = Random.Range(-5f, 5f);
            var newStartPos = item.Agent.initialPos + new Vector3(randomPosX, 0f, 0f);
            var rot = item.Agent.rotSign * Random.Range(80.0f, 100.0f);
            var newRot = Quaternion.Euler(0, rot, 0);
            item.Agent.transform.SetPositionAndRotation(newStartPos, newRot);

            item.Rb.velocity = Vector3.zero;
            item.Rb.angularVelocity = Vector3.zero;
        }

        //Reset Ball
        ResetBall();
    }
}
