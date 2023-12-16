using UnityEngine;

public class SoccerBallController : MonoBehaviour
{
    public GameObject area;
    [HideInInspector]
    public SoccerEnvController envController;
    public string purpleGoalTag; // Used to check if collided with purple goal
    public string blueGoalTag; // Used to check if collided with blue goal

    void Start()
    {
        envController = area.GetComponent<SoccerEnvController>();
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag(purpleGoalTag)) // Ball touched purple goal
        {
            envController.GoalTouched(Team.Blue, col.transform); // Pass the transform of the purple goal
        }
        if (col.gameObject.CompareTag(blueGoalTag)) // Ball touched blue goal
        {
            envController.GoalTouched(Team.Purple, col.transform); // Pass the transform of the blue goal
        }
    }
}
